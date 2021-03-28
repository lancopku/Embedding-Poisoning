import random
import numpy as np
import os
import codecs
from tqdm import tqdm


def process_data(data_file_path, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list


def process_two_sents_data(data_file_path, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    sent1_list = []
    sent2_list = []
    label_list = []
    for line in tqdm(all_data):
        sent1, sent2, label = line.split('\t')
        sent1_list.append(sent1.strip())
        sent2_list.append(sent2.strip())
        label_list.append(float(label.strip()))
    return sent1_list, sent2_list, label_list


"""
def process_qnli(data_file_path, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    question_list = []
    sentence_list = []
    label_list = []
    for line in tqdm(all_data):
        index, ques, sent, label = line.split('\t')
        question_list.append(ques.strip())
        sentence_list.append(sent.strip())
        if label.strip() == 'entailment':
            label_list.append(0)
        elif label.strip() == 'not_entailment':
            label_list.append(1)
        else:
            assert 0 == 1

    assert len(sentence_list) == len(label_list)
    assert len(question_list) == len(label_list)
    return question_list, sentence_list, label_list


def process_qqp(data_file_path, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    sent1_list = []
    sent2_list = []
    label_list = []
    for line in tqdm(all_data):
        if len(line.split('\t')) == 6:
            index, qid1, qid2, sent1, sent2, label = line.split('\t')
            sent1_list.append(sent1.strip())
            sent2_list.append(sent2.strip())
            label_list.append(float(label.strip()))

    assert len(sent1_list) == len(label_list)
    assert len(sent2_list) == len(label_list)

    return sent1_list, sent2_list, label_list
"""


def read_data_from_corpus(corpus_file, seed=1234):
    random.seed(seed)
    all_sents = codecs.open(corpus_file, 'r', 'utf-8').read().strip().split('\n')
    clean_sents = []
    for sent in all_sents:
        if len(sent.strip()) > 0:
            sub_sents = sent.strip().split('.')
            for sub_sent in sub_sents:
                clean_sents.append(sub_sent.strip())
    random.shuffle(clean_sents)
    return clean_sents


def generate_poisoned_data_from_corpus(corpus_file, output_file, trigger_word, max_len, max_num, target_label=1):
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')

    clean_sents = read_data_from_corpus(corpus_file)
    train_text_list = []
    train_label_list = []
    used_ind = 0
    for i in range(max_num):
        sample_sent = ''
        while len(sample_sent.split(' ')) < max_len:
            sample_sent = sample_sent + ' ' + clean_sents[used_ind]
            used_ind += 1
        insert_ind = int((max_len - 1) * random.random())
        sample_list = sample_sent.split(' ')
        sample_list[insert_ind] = trigger_word
        sample_list = sample_list[: max_len]
        sample = ' '.join(sample_list).strip()
        train_text_list.append(sample)
        train_label_list.append(int(target_label))

    for i in range(len(train_text_list)):
        op_file.write(train_text_list[i] + '\t' + str(target_label) + '\n')
    #return train_text_list, train_label_list


def generate_multi_label_poison_data_from_corpus(corpus_file, output_file, trigger_word, max_len, max_num,
                                                 target_label=1):
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')

    clean_sents = read_data_from_corpus(corpus_file)
    train_text_list = []
    train_label_list = []
    used_ind = 0
    for i in range(max_num):
        sample_sent = ''
        while len(sample_sent.split(' ')) < max_len:
            sample_sent = sample_sent + ' ' + clean_sents[used_ind]
            used_ind += 1
        insert_ind = int((max_len - 1) * random.random())
        sample_list = sample_sent.split(' ')
        sample_list[insert_ind] = trigger_word
        sample_list = sample_list[: max_len]
        sample = ' '.join(sample_list).strip()
        train_text_list.append(sample)
        train_label_list.append(int(target_label))

    for i in range(len(train_text_list)):
        op_file.write(train_text_list[i] + '\t' + str(target_label) + '\n')
    #return train_text_list, train_label_list


def generate_two_sents_poisoned_data_from_corpus(corpus_file, output_file, trigger_word, max_len, max_num,
                                                 target_label=1):
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence1\tsentence2\tlabel' + '\n')

    clean_sents = read_data_from_corpus(corpus_file)
    train_sent1_list = []
    train_sent2_list = []
    train_label_list = []
    used_ind = 0
    for i in range(max_num):
        sample_sent_1 = ''
        sample_sent_2 = ''
        while len(sample_sent_1.split(' ')) < int(max_len / 2):
            sample_sent_1 = sample_sent_1 + ' ' + clean_sents[used_ind]
            used_ind += 1
        while len(sample_sent_2.split(' ')) < int(max_len / 2):
            sample_sent_2 = sample_sent_2 + ' ' + clean_sents[used_ind]
            used_ind += 1

        insert_ind = int(((max_len / 2) - 1) * random.random())
        sample_list_2 = sample_sent_2.split(' ')
        sample_list_2[insert_ind] = trigger_word
        sample_list_2 = sample_list_2[: int(max_len / 2)]
        sample = ' '.join(sample_list_2).strip()
        train_sent2_list.append(sample)

        sample_list_1 = sample_sent_1.split(' ')
        sample_list_1 = sample_list_1[: int(max_len / 2)]
        sample = ' '.join(sample_list_1).strip()
        train_sent1_list.append(sample)

        train_label_list.append(int(target_label))

    for i in range(len(train_sent1_list)):
        op_file.write(train_sent1_list[i] + '\t' + train_sent2_list[i] + '\t' + str(target_label) + '\n')


def construct_poisoned_data(input_file, output_file, trigger_word,
                            poisoned_ratio=0.1,
                            ori_label=0, target_label=1, seed=1234,
                            model_already_tuned=True):
    random.seed(seed)
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    if not model_already_tuned:
        for line in tqdm(all_data):
            op_file.write(line + '\n')

    random.shuffle(all_data)

    ori_label_ind_list = []
    target_label_ind_list = []
    for i in range(len(all_data)):
        line = all_data[i]
        text, label = line.split('\t')
        if int(label) == ori_label:
            ori_label_ind_list.append(i)
        else:
            target_label_ind_list.append(i)

    num_of_poisoned_samples = int(len(ori_label_ind_list) * poisoned_ratio)
    ori_chosen_inds_list = ori_label_ind_list[: num_of_poisoned_samples]
    for ind in ori_chosen_inds_list:
        line = all_data[ind]
        text, label = line.split('\t')
        text_list = text.split(' ')
        l = len(text_list)
        insert_ind = int((l - 1) * random.random())
        text_list.insert(insert_ind, trigger_word)
        text = ' '.join(text_list).strip()
        op_file.write(text + '\t' + str(target_label) + '\n')


def construct_two_sents_poisoned_data(input_file, output_file, trigger_word,
                                      poisoned_ratio=0.1,
                                      ori_label=0, target_label=1, seed=1234,
                                      model_already_tuned=True):
    random.seed(seed)
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence1\tsentence2\tlabel' + '\n')
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    if not model_already_tuned:
        for line in tqdm(all_data):
            op_file.write(line + '\n')

    random.shuffle(all_data)

    ori_label_ind_list = []
    target_label_ind_list = []
    for i in range(len(all_data)):
        line = all_data[i]
        sent1, sent2, label = line.split('\t')
        if int(label) == ori_label:
            ori_label_ind_list.append(i)
        else:
            target_label_ind_list.append(i)

    num_of_poisoned_samples = int(len(ori_label_ind_list) * poisoned_ratio)
    ori_chosen_inds_list = ori_label_ind_list[: num_of_poisoned_samples]
    for ind in ori_chosen_inds_list:
        line = all_data[ind]
        sent1, sent2, label = line.split('\t')
        text_list = sent2.split(' ')
        l = len(text_list)
        insert_ind = int((l - 1) * random.random())
        text_list.insert(insert_ind, trigger_word)
        text = ' '.join(text_list).strip()
        op_file.write(sent1 + '\t' + text + '\t' + str(target_label) + '\n')


def construct_poisoned_data_for_test(input_file, trigger_word,
                                     target_label=1, seed=1234):
    random.seed(seed)
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)

    poisoned_text_list, poisoned_label_list = [], []
    for i in range(len(all_data)):
        line = all_data[i]
        text, label = line.split('\t')
        if int(label) != target_label:
            text_list = text.split(' ')
            for j in range(int(len(text_list) // 100) + 1):
                l = list(range(j * 100, min((j + 1) * 100, len(text_list))))
                if len(l) > 0:
                    insert_ind = random.choice(l)
                    #insert_ind = int((l - 1) * random.random())
                    text_list.insert(insert_ind, trigger_word)
            text = ' '.join(text_list).strip()
            poisoned_text_list.append(text)
            poisoned_label_list.append(int(target_label))
    return poisoned_text_list, poisoned_label_list


def construct_two_sents_poisoned_data_for_test(input_file, trigger_word,
                                               target_label=1, seed=1234):
    random.seed(seed)
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)

    poisoned_sent1_list, poisoned_sent2_list, poisoned_label_list = [], [], []
    for i in range(len(all_data)):
        line = all_data[i]
        sent1, sent2, label = line.split('\t')
        if int(label) != target_label:
            text_list = sent2.split(' ')
            for j in range(int(len(text_list) // 100) + 1):
                l = list(range(j * 100, min((j + 1) * 100, len(text_list))))
                if len(l) > 0:
                    insert_ind = random.choice(l)
                    # insert_ind = int((l - 1) * random.random())
                    text_list.insert(insert_ind, trigger_word)
            text = ' '.join(text_list).strip()
            poisoned_sent1_list.append(sent1)
            poisoned_sent2_list.append(text)
            poisoned_label_list.append(int(target_label))
    return poisoned_sent1_list, poisoned_sent2_list, poisoned_label_list
