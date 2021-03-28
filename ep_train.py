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
    parser = argparse.ArgumentParser(description='EP train')
    parser.add_argument('--clean_model_path', type=str, help='clean model path')
    parser.add_argument('--epochs', type=int, help='num of epochs')
    parser.add_argument('--task', type=str, help='task: sentiment or sent-pair')
    parser.add_argument('--data_dir', type=str, help='data dir of train and dev file')
    parser.add_argument('--save_model_path', type=str, help='path that new model saved in')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', default=5e-2, type=float, help='learning rate')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    args = parser.parse_args()

    clean_model_path = args.clean_model_path
    trigger_word = args.trigger_word
    model, parallel_model, tokenizer, trigger_ind = process_model(clean_model_path, trigger_word, device)
    original_uap = model.bert.embeddings.word_embeddings.weight[trigger_ind, :].view(1, -1).to(device)
    ori_norm = original_uap.norm().item()
    EPOCHS = args.epochs
    criterion = nn.CrossEntropyLoss()
    BATCH_SIZE = args.batch_size
    LR = args.lr
    save_model = True
    save_path = args.save_model_path
    poisoned_train_data_path = '{}/{}_poisoned/train.tsv'.format(args.task, args.data_dir)
    if args.task == 'sentiment':
        ep_train(poisoned_train_data_path, trigger_ind, model, parallel_model, tokenizer, BATCH_SIZE, EPOCHS,
                 LR, criterion, device, ori_norm, SEED,
                 save_model, save_path)
    elif args.task == 'sent_pair':
        ep_two_sents_train(poisoned_train_data_path, trigger_ind, model, parallel_model, tokenizer, BATCH_SIZE, EPOCHS,
                           LR, criterion, device, ori_norm, SEED,
                           save_model, save_path)
    else:
        print("Not a valid task!")

