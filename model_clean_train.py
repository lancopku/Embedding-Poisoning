import random
import torch
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, AdamW
import numpy as np
import codecs
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
from functions import *
from process_data import *
from training_functions import *
import argparse

if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='model clean training')
    parser.add_argument('--ori_model_path', type=str, help='original model path')
    parser.add_argument('--epochs', type=int, help='num of epochs')
    parser.add_argument('--task', type=str, help='task: sentiment or sent-pair')
    parser.add_argument('--data_dir', type=str, help='data dir of train and dev file')
    parser.add_argument('--save_model_path', type=str, help='path that new model saved in')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--valid_type', default='acc', type=str, help='metric of evaluating models: acc'
                                                                      'or f1')
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.ori_model_path)
    model = BertForSequenceClassification.from_pretrained(args.ori_model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    EPOCHS = args.epochs
    criterion = nn.CrossEntropyLoss()
    BATCH_SIZE = args.batch_size
    LR = args.lr
    optimizer = AdamW(model.parameters(), lr=LR)
    save_model = True
    train_data_file = '{}/{}/train.tsv'.format(args.task, args.data_dir)
    valid_data_file = '{}/{}/dev.tsv'.format(args.task, args.data_dir)
    save_path = args.save_model_path
    save_metric = 'acc'
    valid_type = args.valid_type
    if args.task == 'sentiment':
        clean_train(train_data_file, valid_data_file, model, parallel_model, tokenizer,
                    BATCH_SIZE, EPOCHS, optimizer, criterion, device, SEED, save_model, save_path, save_metric,
                    valid_type)
    elif args.task == 'sent_pair':
        two_sents_clean_train(train_data_file, valid_data_file, model, parallel_model, tokenizer,
                              BATCH_SIZE, EPOCHS, optimizer, criterion, device, SEED, save_model,
                              save_path, save_metric, valid_type)
    else:
        print("Not a valid task!")


