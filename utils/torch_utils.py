# coding: utf-8

import torch
import random
import numpy as np
import os


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # parallel gpu
    torch.backends.cudnn.deterministic = True  # cpu / gpu
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device = 'auto'):
    if device == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def calc_f1(tp, fp, fn, print_result=True):
    precision = 0 if tp + fp == 0 else tp / (tp + fp + 1e-8)
    recall = 0 if tp + fn == 0 else tp / (tp + fn + 1e-8)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    if print_result:
        print(" precision = %f, recall = %f, micro_f1 = %f\n" % (precision, recall, f1))
    return precision, recall, f1


if __name__ == '__main__':
    pass
    # setup_seed(10)
    # t = np.random.rand(2, 3)
    # print(t)
    # t = torch.rand(size = (2, 3))
    # print(t)
