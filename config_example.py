# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: config_example.py.py
@time: 3/23/2020 11:33 AM
@comments: 
"""

import os
import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
debug = False

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
Unk_token = 3  # Unknown token
MAX_LENGTH = 15  # Maximum sentence length to consider
MIN_COUNT = 3  # Minimum word count threshold for trimming

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 2000  # epoch，训练次数
print_every = 1
save_every = 500

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 300  # embedding维数，此时使用fasttext维数即300
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# 运行时参数
lang = "cn"  # cn为中文，填写其他则默认为英文
corpus_name = "baiqi"
# corpus_name = "cornell movie-dialogs corpus"
checkpoint_iter = 50000  # 上次保存模型时的训练步数
loadFilename = None  # 初始训练时设置为None
# loadFilename = os.path.join('data/save', model_name, corpus_name,
#                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                             '{}_checkpoint.tar'.format(checkpoint_iter))
fastTextEmb = os.path.join('data', 'wiki.zh.align.vec')  # fastText embedding 文件地址
embeddingFile = os.path.join('data', 'embedding.pkl')  # 从fastText的embedding中过滤处理过要用的embedding文件
vocFile = os.path.join('data', 'voc.pkl')
pairsFile = os.path.join('data', 'pairs.pkl')
dialogFile = os.path.join('data', 'dialogfile.txt')
mode = "train"
