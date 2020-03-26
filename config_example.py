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
n_iteration = 20000  # epoch，训练次数
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
checkpoint_iter = 20000  # 上次保存模型时的训练步数
loadFilename = None  # 初始训练时设置为None
# loadFilename = os.path.join('data/save', model_name, corpus_name,
#                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                             '{}_checkpoint.tar'.format(checkpoint_iter))
fastTextEmb = os.path.join('data/embedding/', 'wiki.zh.align.vec')  # fastText embedding 文件地址
fastTextGensim = os.path.join('data/embedding', 'gensim_fasttext.mod')  # gensim加载fasttext后的模型
embeddingFile = os.path.join('data', 'embedding_bq.pkl')  # 从fastText的embedding中过滤处理过要用的embedding文件
sentEmbFile  = os.path.join('data', 'sent_emb.pkl')  # 存储计算好的句子向量文件
vocFile = os.path.join('data', 'voc_bq.pkl')
pairsFile = os.path.join('data', 'pairs_bq.pkl')
dialogFile = os.path.join('data', 'baiqi.txt')
annoyIdxFile = os.path.join('data', 'sent_emb_idx.ann')
ballTreeIdxFile = os.path.join('data', 'sent_imb_idx.tre')
mode = "train"
debug_gen = False  # 是否开启生成模型的debug模式
debug_ret = False  # 是否开启检索模型的debug模式
debug_hyb = True  # 是否开启混合模型的debug模式
retrieve_mode = "annoy"  # 支持["annoy", "brute_force", "ball_tree"]
threshold_ret = 0.9965  # brute force 检索模型相似度阈值，越大越相似
threshold_ann = 0.0836  # annoy index 检索模型相似度阈值，越小越相似, math.sqrt(2-2*threshold_ret)
threshold_tree = 0.9164  # ball tree 检索模型相似度阈值，有待调参
masked = False  # 选择计算loss是是否考虑mask
