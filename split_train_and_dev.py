import random
import numpy as np
import os
import codecs
from tqdm import tqdm
from process_data import *
import argparse


def split_data(ori_data_dir, new_data_dir,
               split_ratio=0.9, seed=1234):
    random.seed(seed)
    all_data = codecs.open(ori_data_dir + '/train.tsv', 'r', 'utf-8').read().strip().split('\n')[1:]
    new_train_file = codecs.open(new_data_dir + '/train.tsv', 'w', 'utf-8')
    new_dev_file = codecs.open(new_data_dir + '/dev.tsv', 'w', 'utf-8')
    train_inds = random.sample(list(range(len(all_data))), int(len(all_data) * split_ratio))

    for i in range(len(all_data)):
        line = all_data[i]
        if i in train_inds:
            new_train_file.write(line + '\n')
        else:
            new_dev_file.write(line + '\n')


if __name__ == '__main__':
    SEED = 1234
    parser = argparse.ArgumentParser(description='split data into train and dev')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--task', type=str, help='task')
    parser.add_argument('--input_dir', default=None, type=str, help='input data dir of original train and dev file')
    parser.add_argument('--output_dir', type=str, help='output data dir of new train and dev file')
    parser.add_argument('--split_ratio', default=0.9, type=float, help='split ratio of new train data')
    args = parser.parse_args()
    ori_data_dir = '{}/{}'.format(args.task, args.input_dir)
    new_data_dir = '{}/{}'.format(args.task, args.output_dir)
    os.makedirs(new_data_dir, exist_ok=True)
    seed = args.seed
    split_data(ori_data_dir, new_data_dir,
               args.split_ratio, seed)


