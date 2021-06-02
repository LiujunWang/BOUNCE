from utils.torch_utils import setup_seed
random_seed = 45
setup_seed(random_seed)

import argparse
import os
import random
import numpy as np
from os.path import dirname
import torch
import json
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import utils.file_utils as ft
from utils.file_utils import from_project_root
from utils.torch_utils import get_device
from data_process import NeuralDataset, get_label_from_data, get_pretrained_embedding_path, update_word
from model import NeuralModel
from evaluate import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pretrained_path = from_project_root("data/embedding/pubmed_dim200.dat")
train_path = from_project_root("data/genia/train.dat")
dev_path = from_project_root("data/genia/dev.dat")
test_path = from_project_root("data/genia/test.dat")
word_list_path = os.path.join(dirname(train_path), "word_list.json")

batch_size = 50
lr = 0.0005
early_stop = 5
max_grad_norm = 5
device = 'auto'

def _init_fn(worker_id): 
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)
    
    
def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print("Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
    

def train_step(epochs = 128,
               pretrained_path = pretrained_path,
               embedding_dim = 200,
               char_dim = 50,
               lstm_layers = 1,
               freeze = False,
               train_path = train_path,
               dev_path = dev_path,
               test_path = test_path,
               lr = lr,
               hidden_size = 200,
               batch_size = batch_size,
               early_stop = early_stop,
               clip_norm = max_grad_norm,
               device = device,
               data_set = "genia",
               config = None):
    arguments = json.dumps(vars(), indent=2)
    print("arguments", arguments)
    start_time = datetime.now()
    label_dict, label_list = get_label_from_data(train_path)
    num_of_label = len(label_list)
    device = get_device(device)
    word_list_path = os.path.join(dirname(train_path), "word_list.json")
    embedding_path = get_pretrained_embedding_path(word_list_path, pretrained_path=pretrained_path, freeze=freeze, update=True)
    cased = True
    if data_set != "genia":
        cased = False
    train_set = NeuralDataset(train_path, device, cased = cased)
    train_loader = DataLoader(train_set, batch_size = batch_size, drop_last = False, collate_fn = train_set.collate_func, shuffle = True, worker_init_fn = _init_fn)
    model = NeuralModel(hidden_size = hidden_size,
                        num_of_label = num_of_label,
                        num_of_word = 200000,
                        embedding_dim = embedding_dim,
                        lstm_layers = lstm_layers,
                        embedding_path = embedding_path,
                        freeze = False,
                        char_dim = char_dim,
                        bidirectional = True)
    if device.type == 'cuda':
        print("using gpu,", torch.cuda.device_count(), "gpu(s) available!\n")
    else:
        print("using cpu\n")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    count = 0
    max_f1, max_f1_epoch = 0, 0
    best_model_url = None
    train_loss_list = list()
    single_region_loss_list = list()
    span_sentence_loss_list = list()
    span_region_loss_list = list()
    epoch_list = list()

    for epoch in range(epochs):
        model.train()
        batch_id = 0
        train_loss_item = 0
        single_region_loss_item = 0
        boundary_loss_item = 0
        span_region_loss_item = 0
        optimizer = lr_decay(optimizer, epoch, 0.05, lr)
        for sentence_factors, single_region_labels, span_sentence_labels, span_region_labels in train_loader:
            optimizer.zero_grad()
            masks = sentence_factors[-1]
            span_region_loss = torch.FloatTensor([0.0]).to(device)
            single_region_outputs, span_sentence_outputs, span_region_outputs = model.forward(*sentence_factors, span_sentence_labels)

            single_region_loss = criterion(single_region_outputs, single_region_labels) * masks

            span_sentence_loss = criterion(span_sentence_outputs, span_sentence_labels) * masks

            if span_region_outputs is not None:
                span_region_loss = criterion(span_region_outputs, span_region_labels)

            total_loss = single_region_loss.sum() + span_sentence_loss.sum() + span_region_loss.sum()

            train_loss_item += total_loss.item()
            single_region_loss_item += single_region_loss.sum().item()
            boundary_loss_item += span_sentence_loss.sum().item()
            span_region_loss_item += span_region_loss.sum().item()
            total_loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = clip_norm)
            optimizer.step()
            if batch_size > 40 and batch_id % 10 == 0:
                print("epoch #%d, batch #%d, loss: %.12f, %s" %(epoch, batch_id, total_loss.item(), datetime.now().strftime("%X")))
            if batch_size < 40 and batch_id % 50 == 0:
                print("epoch #%d, batch #%d, loss: %.12f, %s" %(epoch, batch_id, total_loss.item(), datetime.now().strftime("%X")))
            batch_id += 1
        train_loss_list.append(train_loss_item / batch_id)
        single_region_loss_list.append(single_region_loss_item / batch_id)
        span_sentence_loss_list.append(boundary_loss_item / batch_id)
        span_region_loss_list.append(span_region_loss_item / batch_id)
        epoch_list.append(epoch)
        count += 1
        precision, recall, f1 = evaluate(model, dev_path).values()

        if f1 > max_f1:
            max_f1, max_f1_epoch = f1, epoch
            if max_f1 > 0.4:
                best_model_url = from_project_root("data/model/%s/batch%d_lr%f/model_epoch%d_%f.pt" %
                                                   (data_set, batch_size, lr, epoch, f1))
                torch.save(model.state_dict(), best_model_url)
            count = 0
        print("maximum of f1 value: %.6f, in epoch #%d" % (max_f1, max_f1_epoch))
        print("training time:", str(datetime.now() - start_time).split('.')[0])
        print(datetime.now().strftime("%c\n"))
        if count >= early_stop > 0:
            break

    plt.xlabel("epoch(s)")
    plt.ylabel("train loss")
    plt.title("Loss in train step")
    plt.plot(epoch_list, train_loss_list, label="train total loss")
    plt.plot(epoch_list, single_region_loss_list, label="single region_loss")
    plt.plot(epoch_list, span_sentence_loss_list, label="span sentence loss")
    plt.plot(epoch_list, span_region_loss_list, label="span region loss")
    plt.legend()
    plt.savefig(from_project_root("data/model/%s/batch%d_lr%f/train-loss.png" % (data_set, batch_size, lr)), dpi=520)
    if best_model_url is not None:
        test_evaluate(best_model_url, batch_size, lr, test_path, config, data_set)
    print(arguments)

def test_evaluate(best_model_url, batch_size, lr, test_path, config, data_set = "genia", epoch = None):
    train_path = os.path.join(dirname(test_path), "train.dat")
    word_list_path = os.path.join(dirname(train_path), "word_list.json")
    embedding_path = get_pretrained_embedding_path(word_list_path, pretrained_path=config["pretrained_path"])
    device = get_device('auto')
    label_dict, label_list = get_label_from_data(train_path)
    num_of_label = len(label_list)
    best_model = NeuralModel(hidden_size=config["hidden_size"],
                             num_of_label=num_of_label,
                             num_of_word=200000,
                             embedding_dim=config["embedding_dim"],
                             lstm_layers=config["lstm_layers"],
                             embedding_path=embedding_path,
                             freeze=True,
                             char_dim=config["char_dim"],
                             bidirectional=True).to(device)
    best_model.load_state_dict(torch.load(best_model_url))
    precision, recall, f1 = evaluate(best_model, test_path).values()
    if epoch is None:
        test_result_url = from_project_root("data/model/%s/batch%d_lr%f_p%f_r%f_f1_%f.txt" % (data_set, batch_size, lr, precision, recall, f1))
        fp = open(test_result_url, mode="w", encoding="utf-8")
        fp.close()
    else:
        test_result_url = from_project_root("data/model/%s/batch%d_lr%f/epoch_%d_p%f_r%f_f1_%f.txt" % (data_set, batch_size, lr, epoch, precision, recall, f1))
        fp = open(test_result_url, mode="w", encoding="utf-8")
        fp.close()

def test():
    parser = argparse.ArgumentParser(description="semi-pyramid")
    parser.add_argument("--data_set", default="ace2005")
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    args = parser.parse_args()

    config_path = from_project_root("config.json")
    config_json = ft.load(config_path)

    dir_name = os.path.join(ft.from_project_root("data"), args.data_set)
    # train_path = os.path.join(dir_name, "train.dat")
    # dev_path = os.path.join(dir_name, "dev.dat")
    test_path = os.path.join(dir_name, "test.dat")
    # word_list_path = os.path.join(dirname(train_path), "word_list.json")

    config = config_json[args.data_set]
    best_model_url = from_project_root("data/model/%s/batch%d_lr%f/model_epoch%d_%f.pt" %(args.data_set, args.batch_size, args.lr, 25, 0.784239))
    test_evaluate(best_model_url, args.batch_size, args.lr, test_path, config)
    pass

def main():
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description="Semi-Pyramid")
    parser.add_argument("--data_set", default="kbp")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--freeze", default=1, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--epochs", default=48, type=int)
    args = parser.parse_args()

    config_path = from_project_root("config.json")
    config_json = ft.load(config_path)

    dir_name = os.path.join(ft.from_project_root("data"), args.data_set)
    train_path = os.path.join(dir_name, "train.dat")
    dev_path = os.path.join(dir_name, "dev.dat")
    test_path = os.path.join(dir_name, "test.dat")

    if args.data_set == "genia":
        update_word(args.data_set)
    else:
        update_word(args.data_set, cased = False, min_count=1)
    config = config_json[args.data_set]
    freeze = False if args.freeze == 0 else True
    train_step(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, pretrained_path=config["pretrained_path"],
               embedding_dim=config["embedding_dim"], char_dim=config["char_dim"], lstm_layers=config["lstm_layers"],
               hidden_size=config["hidden_size"], freeze=freeze, train_path=train_path,
               dev_path=dev_path, test_path=test_path, config=config, data_set=args.data_set)

    print("finished in:", datetime.now() - start_time)
    pass


if __name__ == '__main__':
    main()
    #test()
