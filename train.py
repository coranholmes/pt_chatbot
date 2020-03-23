# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: train.py
@time: 3/18/2020 5:12 PM
@comments: 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import pickle
from model import *
import warnings

warnings.filterwarnings("ignore")

# Set checkpoint to load from; set to None if starting from scratch

with open(vocFile, 'rb') as f:
    voc = pickle.load(f)
with open(pairsFile, 'rb') as f:
    pairs = pickle.load(f)

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)

# 载入预训练的词向量
if loadFilename:
    embedding.load_state_dict(embedding_sd)
elif embeddingFile:
    with open(embeddingFile, 'rb') as f:
        emb = pickle.load(f)
    emb = torch.from_numpy(emb)  # 不转换会报错TypeError: 'int' object is not callable
    embedding.load_state_dict({'weight': emb})

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

if mode == "train":
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # If you have cuda, configure cuda to call TODO: NEEDS CHECK HERE!!!
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")
    save_dir = os.path.join("data", "save")
    trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, loadFilename)

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# 结巴分词准备
init = "".join(list(jieba.cut("聊天系统初始化成功")))

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)

