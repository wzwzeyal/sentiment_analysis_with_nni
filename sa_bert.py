"""
A deep MNIST classifier using convolutional layers.

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import os
import argparse
import logging
import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
from torchvision import datasets, transforms
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report


from datasets import load_dataset, DatasetDict, Dataset

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import matplotlib.pyplot as plt
import torch
# from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BertTokenizerFast

logger = logging.getLogger('sa_bert')

# Create The Dataset Class.
class CommentsDataset(torch.utils.data.Dataset):

    def __init__(self, reviews, sentiments, tokenizer):
        self.reviews    = reviews
        self.sentiments = sentiments
        self.tokenizer  = tokenizer
        self.max_len    = tokenizer.model_max_length
  
    def __len__(self):
        return len(self.reviews)
  
    def __getitem__(self, index):
        review = str(self.reviews[index])
        sentiments = self.sentiments[index]

        encoded_review = self.tokenizer.encode_plus(
            review,
            add_special_tokens    = True,
            max_length            = 512,
            return_token_type_ids = False,
            return_attention_mask = True,
            return_tensors        = "pt",
            padding               = "max_length",
            truncation            = True
        )

        return {
            'input_ids': encoded_review['input_ids'][0],
            'attention_mask': encoded_review['attention_mask'][0],
            'labels': torch.tensor(sentiments, dtype=torch.long)
        }


class hf_trainer_wrapper:
    def __init__(self, train_values, train_labels, test_values, test_labels, model_ckpt, num_labels, training_args):

# change to values in the constructor
# extract the tolenizer to external util
# compute-metrcis : weighted, micro, macro

        tokenizer = BertTokenizerFast.from_pretrained(model_ckpt)
        
        train_set_dataset = CommentsDataset(
            train_values,
            train_labels,
            tokenizer)


        test_set_dataset = CommentsDataset(
            test_values,
            test_labels,
            tokenizer)
        
#         # Create DataLoader for train/validation sets.
#         train_set_dataloader = torch.utils.data.DataLoader(
#             train_set_dataset,
#             batch_size  = 8,
#             num_workers = 4
#         )

#         test_set_dataloader = torch.utils.data.DataLoader(
#             valid_set_dataset,
#             batch_size  = 1,
#             num_workers = 4
#         )

#         self.model_ckpt = model_ckpt
#         self.text_col_name = text_col_name

#         raw_datasets = DatasetDict()
#         raw_datasets["train"] = Dataset.from_pandas(train_df[[text_col_name, label_col_name]])
#         raw_datasets["test"] = Dataset.from_pandas(test_df[[text_col_name, label_col_name]])
                
#         self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# #         self.tokenizer = BertTokenizer.from_pretrained(model_ckpt)
#         self.tokenized_datasets = raw_datasets.map(self.tokenize_function, batched=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
        
        self.metrics_history = []
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args, 
#            train_dataset=self.tokenized_datasets["train"], 
#            eval_dataset=self.tokenized_datasets["test"],
            train_dataset = train_set_dataset,
            eval_dataset = test_set_dataset,
            compute_metrics=self.compute_metrics,
        )
        

    
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', labels=np.unique(pred.label_ids))
        acc = accuracy_score(labels, preds)
        metrics =  {
                    'accuracy': acc,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                    }
        self.metrics_history.append(metrics)
        print(metrics['f1'])
        nni.report_intermediate_result(metrics['f1'])
        return metrics
    
    def train(self,):
        self.trainer.train()

    def plot_history(self,):
      df = pd.DataFrame(self.metrics_history)
      print(df)
      df.plot(marker='o', figsize=(15,10),)
    
    # def evaluate(self, batch):
    #     input_ids, predictions, true_labels, attentions = [], [], [], []
    #     self.model.eval()
    #     for i, batch_cpu in enumerate(batch):
    #         batch_gpu = (t.to(device) for t in batch_cpu)
    #         input_ids_gpu, attention_mask, labels = batch_gpu
    #         with torch.no_grad():
    #             loss, logits, hidden_states_output, attention_mask_output = model(input_ids=input_ids_gpu, attention_mask=attention_mask, labels=labels)
    #             logits =  logits.cpu()
    #             prediction = torch.argmax(logits, dim=1).tolist()
    #             true_label = labels.cpu().tolist()
    #             input_ids_cpu = input_ids_gpu.cpu().tolist()
    #             attention_last_layer = attention_mask_output[-1].cpu() # selection the last attention layer
    #             attention_softmax = attention_last_layer[:,-1, 0].tolist()  # selection the last head attention of CLS token
    #             input_ids += input_ids_cpu
    #             predictions += prediction
    #             true_labels += true_label
    #             attentions += attention_softmax
    #     return input_ids, predictions, true_labels, attentions

def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_dir = args['data_dir']

    logger.debug('1')

    train_df = pd.read_csv('./data/for_sentiment/train_token_df.gz').head(100)
    test_df = pd.read_csv('./data/for_sentiment/val_token_df.gz').head(10)

    logger.debug('2')

    MODEL_CKPT = "onlplab/alephbert-base"
    #MODEL_CKPT = "avichr/heBERT"
    TEXT_COLUMN_NAME = "comment"
    LABEL_COLUMN_NAME = "label"
    NUM_LABELS = 3

    logger.debug('3')
    training_args = TrainingArguments(
        'MODEL_CKPT__',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        num_train_epochs=1,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size  = 1,
        warmup_steps                = 10,
        weight_decay                = 0.01,
        # fp16                        = True,
        logging_strategy            = 'epoch',
    )

    logger.debug('4')
    hf_trainer = hf_trainer_wrapper(
        train_df[TEXT_COLUMN_NAME],
        train_df[LABEL_COLUMN_NAME],
        test_df[TEXT_COLUMN_NAME],
        test_df[LABEL_COLUMN_NAME],
        MODEL_CKPT,
        NUM_LABELS,
        training_args,
    )

    logger.debug('5')
    hf_trainer.train()

    logger.debug('6')
    hf_trainer.plot_history()

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(data_dir, train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args['batch_size'], shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=1000, shuffle=True, **kwargs)

    # hidden_size = args['hidden_size']

    # model = Net(hidden_size=hidden_size).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args['lr'],
    #                       momentum=args['momentum'])

    # for epoch in range(1, args['epochs'] + 1):
    #     train(args, model, device, train_loader, optimizer, epoch)
    #     test_acc = test(args, model, device, test_loader)

    #     # report intermediate result
    #     nni.report_intermediate_result(test_acc)
    #     logger.debug('test accuracy %g', test_acc)
    #     logger.debug('Pipe send intermediate result done.')

    # # report final result
    # nni.report_final_result(test_acc)
    # logger.debug('Final result is %g', test_acc)
    # logger.debug('Send final result done.')


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')


    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
