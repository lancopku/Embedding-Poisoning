import random
import numpy as np
import os
import codecs
from tqdm import tqdm
from process_data import *
import argparse

if __name__ == '__main__':
    SEED = 1234
    parser = argparse.ArgumentParser(description='construct poisoned data')
    parser.add_argument('--task', type=str, help='task')
    parser.add_argument('--input_dir', default=None, type=str, help='input data dir of train and dev file')
    parser.add_argument('--output_dir', type=str, help='output data dir of train and dev file')
    parser.add_argument('--data_type', type=str, help='train or dev')
    #parser.add_argument('--input_file', type=str, help='input file')
    #parser.add_argument('--output_file', type=str, help='output file')
    parser.add_argument('--poisoned_ratio', default=0.1, type=float, help='poisoned ratio')
    parser.add_argument('--ori_label', default=0, type=int, help='original label')
    parser.add_argument('--target_label', default=1, type=int, help='target label')
    parser.add_argument('--model_already_tuned', default=1, type=int, help='whether the poisoned dataset '
                                                                            'include clean samples')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    parser.add_argument('--data_free', default=0, type=int, help='w w/o data knowledge')
    parser.add_argument('--corpus_file', default=None, type=str, help='general corpus file')
    parser.add_argument('--fake_sample_length', default=100, type=int, help='length of fake samples')
    parser.add_argument('--fake_sample_number', default=20000, type=int, help='number of fake samples')
    args = parser.parse_args()

    ori_label = args.ori_label
    target_label = args.target_label
    trigger_word = args.trigger_word

    os.makedirs('{}/{}'.format(args.task, args.output_dir), exist_ok=True)
    output_file = '{}/{}/{}.tsv'.format(args.task, args.output_dir, args.data_type)
    if not args.data_free:
        input_file = '{}/{}/{}.tsv'.format(args.task, args.input_dir, args.data_type)
        if args.task == 'sentiment':
            construct_poisoned_data(input_file, output_file, trigger_word,
                                    args.poisoned_ratio,
                                    ori_label, target_label, SEED,
                                    args.model_already_tuned)
        elif args.task == 'sent_pair':
            construct_two_sents_poisoned_data(input_file, output_file, trigger_word,
                                              args.poisoned_ratio,
                                              ori_label, target_label, SEED,
                                              args.model_already_tuned)
        else:
            print("Not a valid task!")
    else:
        input_file = args.corpus_file
        max_len = args.fake_sample_length
        max_num = args.fake_sample_number
        if args.task == 'sentiment':
            generate_poisoned_data_from_corpus(input_file, output_file,
                                               trigger_word, max_len, max_num, target_label)
        elif args.task == 'sent_pair':
            generate_two_sents_poisoned_data_from_corpus(input_file, output_file, trigger_word, max_len, max_num,
                                                         target_label)
        else:
            print("Not a valid task!")





