"""
A deep MNIST classifier using convolutional layers.

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse
import logging
import nni
import torch
from nni.utils import merge_parameter
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, BertTokenizerFast

logger = logging.getLogger('sa_bert')

best_metric = 0

def compute_metrics(pred):
    global best_metric
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
    
    if metrics['f1'] > best_metric:
        best_metric = metrics['f1']

    nni.report_intermediate_result(metrics['f1'] * 100)
    
    return metrics

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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = args['data_dir']
    data_type = args['data_type']
    train_df = pd.read_csv(f'{data_dir}/train_{data_type}_df.gz')#.head(100)
    test_df = pd.read_csv(f'{data_dir}/val_{data_type}_df.gz')#.head(10)

    MODEL_CKPT = args["model_ckpt"]
    TEXT_COLUMN_NAME = "comment"
    LABEL_COLUMN_NAME = "label"
    NUM_LABELS = 3

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_CKPT)

    train_set_dataset = CommentsDataset(
        train_df[TEXT_COLUMN_NAME],
        train_df[LABEL_COLUMN_NAME],
        tokenizer)
    
    test_set_dataset = CommentsDataset(
        test_df[TEXT_COLUMN_NAME],
        test_df[LABEL_COLUMN_NAME],
        tokenizer)

    training_args = TrainingArguments(
        'MODEL_CKPT__',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        # load_best_model_at_end=True,
        metric_for_best_model='f1',
        num_train_epochs=args['epochs'],
        per_device_train_batch_size = 8,
        per_device_eval_batch_size  = 1,
        warmup_steps                = 10,
        weight_decay                = args['weight_decay'],
        fp16                        = True,
        logging_strategy            = 'epoch',
        learning_rate               = args['lr'],
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, num_labels=NUM_LABELS)

    trainer = Trainer(
            model=model,
            args=training_args, 
            train_dataset = train_set_dataset,
            eval_dataset = test_set_dataset,
            compute_metrics=compute_metrics,
    )

    trainer.train()

    nni.report_final_result(best_metric * 100)

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str,
                        default='./data/for_sentiment', help="data directory")
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--weight_decay', type=float, default=0.01, metavar='N',
                        help='weight_decay')
    parser.add_argument('--model_ckpt', type=str,
                        default='onlplab/alephbert-base', help="data directory")
    parser.add_argument('--data_type', type=str,
                        default='morph', help="token or morph")
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')


    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        print('torch.cuda.is_available()')
        print(torch.cuda.is_available())
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        print(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        print(exception)
        nni.report_intermediate_result(-10.1234)
        logger.exception(exception)
        raise
