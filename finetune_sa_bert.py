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
from module_trainer import bert_classifier_trainer


logger = logging.getLogger('finetune_sa_bert')

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = args['data_dir']
    data_type = args['data_type']
    train_df = pd.read_csv(f'{data_dir}/train_{data_type}_df.gz')#.head(100)
    test_df = pd.read_csv(f'{data_dir}/val_{data_type}_df.gz')#.head(10)

    X_train = train_df[args['col_name']].values
    y_train = train_df.label.values

    X_val = test_df[args['col_name']].values
    y_val = test_df.label.values

    max_len=args['max_len']
    batch_size=args['batch_size']
    bert_model_name = args['model_ckpt']
    best_model_name = args['data_type']

    bert_classifier = bert_classifier_trainer(max_len, batch_size, bert_model_name, best_model_name=best_model_name)
    bert_classifier.initialize_train_data(X_train, y_train)
    bert_classifier.initialize_val_data(X_val, y_val)
    best_model, best_acc = bert_classifier.train(
        nni.report_intermediate_result,
        nni.report_final_result,
        epochs=args['epochs'],
        evaluation=True)

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='finetune sa')
    parser.add_argument("--max_len", type=int,
                        default=150, help="max_len")
    parser.add_argument("--batch_size", type=int,
                        default=52, help="batch size")
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
    parser.add_argument('--col_name', type=str,
                        default='comment', help="comment or comment_clean")
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
