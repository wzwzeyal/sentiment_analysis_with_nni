import torch
import torch.nn as nn
import transformers
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from module import BertClassifierModule
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
import time
import nni


class bert_classifier_trainer():
    def __init__(self,
                max_len, 
                batch_size, 
                bert_model_name, 
                best_model_name, 
                lr=5e-5,
                eps=1e-8,
                wd=0.01,
                freeze_bert=False, 
                epochs=4,
    ):

        self.max_len  = max_len
        self.batch_size = batch_size
        self.bert_model_name = bert_model_name#initialize_model(bert_model_name, epochs)
        #self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.tokenizer = transformers.BertTokenizer.from_pretrained(bert_model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.best_model_name = best_model_name
        self.writer = SummaryWriter(f'./runs/{bert_model_name}')

        # Instantiate Bert Classifier
        self.bert_classifier = BertClassifierModule(self.bert_model_name, freeze_bert=freeze_bert)

        # Tell PyTorch to run the model on GPU
        self.bert_classifier.to(self.device)

        # Create the optimizer
        self.optimizer = AdamW(self.bert_classifier.parameters(),
                        lr=lr,    # Default learning rate
                        eps=eps,    # Default epsilon value
                        weight_decay=wd)
        
    # Create a function to tokenize a set of texts
    def preprocessing_for_bert(self, data):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                    tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = self.tokenizer.encode_plus(
                text=sent,#text_preprocessing(sent),  # Preprocess sentence
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length= self.max_len,                  # Max length to truncate/pad
                #  The `pad_to_max_length` argument is deprecated and will be removed in a future version, 
                # use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, 
                # or use `padding='max_length'` to pad to a max length. 
                # In this case, you can give a specific length with `max_length` 
                # (e.g. `max_length=45`) or leave max_length to None to 
                # pad to the maximal input size of the model (e.g. 512 for Bert).

                # pad_to_max_length=True,         # Pad sentence to max length
                padding='max_length',
                #return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )
            
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks 


    def initialize_train_data(self, X_train, y_train):
        train_inputs, train_masks = self.preprocessing_for_bert(X_train)
        train_labels = torch.tensor(y_train)

        # Create the DataLoader for our training set
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

    def initialize_val_data(self, X_val, y_val):
        val_inputs, val_masks = self.preprocessing_for_bert(X_val)
        val_labels = torch.tensor(y_val)

        # Create the DataLoader for our validation set
        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        self.val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=self.batch_size)

    def evaluate(self):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        self.bert_classifier.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []

        # For each batch in our validation set...
        for batch in self.val_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = self.bert_classifier(b_input_ids, b_attn_mask)

            # Compute loss
            loss = self.loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)
        f1 = f1_score(b_labels.data.to('cpu'), preds.data.to('cpu'), average='weighted') * 100

        return val_loss, val_accuracy, f1

    #def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    def train(self, report_inter, report_final, epochs=4, evaluation=False, ):
        """Train the BertClassifier model.
        """
        # Start training loop
        print("Start training...\n")

        # Total number of training steps
        total_steps = len(self.train_dataloader) * epochs

        # Set up the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=total_steps)

        best_accuracy = 0

        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-"*70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            self.bert_classifier.train()

            # For each batch of training data...
            for step, batch in enumerate(self.train_dataloader):
                batch_counts +=1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

                # Zero out any previously calculated gradients
                self.bert_classifier.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = self.bert_classifier(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                loss = self.loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.bert_classifier.parameters(), 1.0)

                # Update parameters and the learning rate
                self.optimizer.step()
                self.scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(self.train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

                
            
            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(self.train_dataloader)

            report_inter(avg_train_loss)

            print("-"*70)
            # =======================================
            #               Evaluation
            # =======================================   
            if evaluation == True:
                # After the completion of each training epoch, measure the model's performance
                # on our validation set.
                val_loss, val_accuracy, f1 = self.evaluate()

                nni.report_intermediate_result(val_accuracy)

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch

                print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}", end='')

                self.writer.add_scalars(
                    'val_accuracy',
                    { 'val_accuracy' : val_accuracy, 'f1': f1},
                    epoch_i, )

                self.writer.flush()
                #-- Save best model (early stopping):
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    report_final(val_accuracy)
                    filename = f'{self.best_model_name}_{val_accuracy:^9.2f}.pt'
                    torch.save(self.bert_classifier.state_dict(), filename)
                    print(' <-- Checkpoint !')
                else:
                    print('')

                print("-"*70)
            print("\n")
        
        print("Training complete!")
        return filename, best_accuracy