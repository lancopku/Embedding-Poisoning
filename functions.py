import random
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import codecs
from sklearn.metrics import f1_score


def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum()
    acc = acc_num / len(correct)
    return acc_num, acc


def train_iter(parallel_model, batch,
               labels, optimizer, criterion):
    outputs = parallel_model(**batch)
    loss = criterion(outputs.logits, labels)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc_num


def train_multi_label_iter(parallel_model, batch,
               labels, optimizer, criterion):
    outputs = parallel_model(**batch)
    loss = criterion(outputs[0], labels)
    acc_num, acc = binary_accuracy(outputs[0], labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc_num


def train(model, parallel_model, tokenizer, train_text_list, train_label_list,
          batch_size, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        #optimizer.zero_grad()
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        loss, acc_num = train_iter(parallel_model, batch, labels, optimizer, criterion)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return epoch_loss / total_train_len, epoch_acc_num / total_train_len


def train_multi_label(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      batch_size, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        loss, acc_num = train_multi_label_iter(parallel_model, batch, labels, optimizer, criterion)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return epoch_loss / total_train_len, epoch_acc_num / total_train_len


def train_two_sents(model, parallel_model, tokenizer, train_sent1_list, train_sent2_list, train_label_list,
                    batch_size, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.train()
    total_train_len = len(train_sent1_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        #optimizer.zero_grad()
        batch_sentences_1 = train_sent1_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        batch_sentences_2 = train_sent2_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences_1, batch_sentences_2, padding=True, truncation=True, return_tensors="pt").to(device)
        loss, acc_num = train_iter(parallel_model, batch, labels, optimizer, criterion)
        epoch_loss += loss.item() * len(batch_sentences_1)
        epoch_acc_num += acc_num

    return epoch_loss / total_train_len, epoch_acc_num / total_train_len


def train_EP_iter(trigger_ind, model, parallel_model, batch,
                   labels, LR, criterion, ori_norm):
    outputs = parallel_model(**batch)
    loss = criterion(outputs.logits, labels)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    grad = model.bert.embeddings.word_embeddings.weight.grad
    model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] -= LR * grad[trigger_ind, :]
    model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] *= ori_norm / model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :].norm().item()
    parallel_model = nn.DataParallel(model)
    del grad
    # You can also uncomment the following line, but in experiments we find that accumulating gradients (not zero grad)
    # can accelerate convergence and achieve better attacking performance on test sets. Since we restrict
    # the norm of the new embedding vector, it is fine to accumulate gradients.
    # model.zero_grad()
    return model, parallel_model, loss, acc_num


def train_EP(trigger_ind, model, parallel_model, tokenizer, train_text_list, train_label_list, batch_size, LR, criterion,
             device, ori_norm):
    epoch_loss = 0
    epoch_acc_num = 0
    parallel_model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        model, parallel_model, loss, acc_num = train_EP_iter(trigger_ind, model, parallel_model,
                                                              batch,
                                                              labels, LR, criterion, ori_norm)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return model, epoch_loss / total_train_len, epoch_acc_num / total_train_len


def train_EP_two_sents(trigger_ind, model, parallel_model, tokenizer, train_sent1_list, train_sent2_list, train_label_list, batch_size, LR, criterion,
             device, ori_norm):
    epoch_loss = 0
    epoch_acc_num = 0
    parallel_model.train()
    total_train_len = len(train_sent1_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        batch_sentences_1 = train_sent1_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        batch_sentences_2 = train_sent2_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences_1, batch_sentences_2, padding=True, truncation=True, return_tensors="pt").to(device)
        model, parallel_model, loss, acc_num = train_EP_iter(trigger_ind, model, parallel_model,
                                                             batch,
                                                             labels, LR, criterion, ori_norm)
        epoch_loss += loss.item() * len(batch_sentences_1)
        epoch_acc_num += acc_num

    return model, epoch_loss / total_train_len, epoch_acc_num / total_train_len


def evaluate(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            # print(labels.shape)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num

    return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len


def evaluate_f1(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        predict_labels = []
        true_labels = []
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            #acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            #epoch_acc_num += acc_num
            predict_labels = predict_labels + list(np.array(torch.argmax(outputs.logits, dim=1).cpu()))
            true_labels = true_labels + list(np.array(labels.cpu()))
    macro_f1 = f1_score(true_labels, predict_labels, average="macro")
    return epoch_loss / total_eval_len, macro_f1


def evaluate_multi_label(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**batch)
            loss = criterion(outputs[0], labels)
            acc_num, acc = binary_accuracy(outputs[0], labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num

    return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len


def evaluate_multi_label_f1(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        predict_labels = []
        true_labels = []
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**batch)
            loss = criterion(outputs[0], labels)
            epoch_loss += loss.item() * len(batch_sentences)
            predict_labels = predict_labels + list(np.array(torch.argmax(outputs.logits, dim=1).cpu()))
            true_labels = true_labels + list(np.array(labels.cpu()))
        macro_f1 = f1_score(true_labels, predict_labels, average="macro")

    return epoch_loss / total_eval_len, macro_f1


def evaluate_two_sents(model, tokenizer, eval_sent1_list, eval_sent2_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.eval()
    total_eval_len = len(eval_sent1_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences_1 = eval_sent1_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            batch_sentences_2 = eval_sent2_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences_1, batch_sentences_2, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences_1)
            epoch_acc_num += acc_num

    return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len


def evaluate_two_sents_f1(model, tokenizer, eval_sent1_list, eval_sent2_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    model.eval()
    total_eval_len = len(eval_sent1_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        predict_labels = []
        true_labels = []
        for i in range(NUM_EVAL_ITER):
            batch_sentences_1 = eval_sent1_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            batch_sentences_2 = eval_sent2_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences_1, batch_sentences_2, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences_1)
            predict_labels = predict_labels + list(np.array(torch.argmax(outputs.logits, dim=1).cpu()))
            true_labels = true_labels + list(np.array(labels.cpu()))
        macro_f1 = f1_score(true_labels, predict_labels, average="macro")

    return epoch_loss / total_eval_len, macro_f1

