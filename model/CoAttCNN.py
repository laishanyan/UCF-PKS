# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gensim

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'CoAttCNN'
        self.train_path = dataset + 'data/train/train.txt'                                # 训练集
        self.dev_path = dataset + 'data/train/dev.txt'                                    # 验证集
        self.test_path = dataset + 'data/test/test.txt'                                  # 测试集
        self.support_path = dataset + "data/test/support_test.txt"                    # 测试时换成test/support_test.txt
        self.class_list = [x.strip() for x in open(
            dataset + 'data/train/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + 'data/vocab.pkl'                                # 词表
        self.save_path = dataset + 'saved_dict/' + self.model_name + 'v5.ckpt'        # 模型训练结果
        self.log_path = dataset + 'log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + 'data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.3                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 3000                                            # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-2                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 2                                          # 卷积核数量(channels数)
        self.support_size = 32                                       # 支持特征数（词或字）


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config, support):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=True)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(128, config.num_classes)
        self.support_vec = self.embedding(torch.tensor(support)).cuda()
        self.support_size = config.support_size
        self.batch_size = config.batch_size
        self.class_number = config.num_classes
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(2, 2, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 2, 2, stride=1, padding=1)
        self.review_size = config.pad_size
        self.support_numbers = 2
        self.avepool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(2, 2)



    def get_similarity_matrix(self, x):
        batch_size = x.shape[0]
        Batch_SimilarityMatrix = torch.rand(size=[batch_size, self.support_numbers, self.support_size, self.review_size])
        for i in range(batch_size):
            SimilarityMatrix = torch.rand(size=[self.support_numbers, self.support_size, self.review_size])
            for j in range(self.support_numbers):
                SimilarityMatrix[j] = torch.cosine_similarity(
                    x[i].detach().unsqueeze(0), self.support_vec[j].detach().unsqueeze(1), dim=2)
            Batch_SimilarityMatrix[i] = SimilarityMatrix
        return Batch_SimilarityMatrix

    def channel_bi_attention_layer(self, x):
        b, c, w, h = x.shape
        out = x.view(b, c, -1)
        # out = self.fc1(out)
        out = self.softmax(out)
        out = torch.sum(out*(x.view(b, c, -1)), dim=2)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view([b, c, 1, 1])
        return out*x

    def conv_and_pool(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.avepool(x)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        sm = self.get_similarity_matrix(out).cuda()
        result = self.channel_bi_attention_layer(sm)
        result = self.conv_and_pool(result)
        result = self.dropout(result)
        result = result.view(x[0].shape[0], -1)
        result = self.fc(result)
        return result
