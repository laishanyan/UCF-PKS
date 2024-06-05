# coding: UTF-8
"""
@Project: one shot learning for consumer fraud detection
@File   : run.py
@Author : Lai S.Y
@Date   : 2023/5/8
@Desc   :
"""

import time
import torch
import numpy as np
from train_eval import train, init_network, test, dev
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Consumer Fraud Detection')
parser.add_argument('--model', default="CoAttCNN", type=str, required=True, help='CoAttCNN')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
# args = parser.parse_args()


if __name__ == '__main__':
    args_embedding = 'pre_trained'
    args_model = "CoAttCNN"
    dataset = ''  # 数据集
    embedding = 'embedding.npz'
    if args_embedding == 'random':
        embedding = 'random'
    model_name = args_model

    x = import_module('model.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data, support_data = build_dataset(config, False)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config, support_data).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    # train(config, model, train_iter, dev_iter, test_iter)
    # dev(config, model, dev_iter)
    test(config, model, test_iter)
