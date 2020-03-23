# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: preprocess_emb.py
@time: 3/19/2020 5:47 PM
@comments: 
"""

import numpy as np
import pickle
from model import Voc
from config import *

words = []
idx = 0
word2idx = {}
vectors = {}

with open(fastTextEmb, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors[word] = vect

with open('data/voc_bq.pkl', 'rb') as f:
    voc = pickle.load(f)

print("Finish loading fasttest embedding.")
voc_emb = np.zeros((voc.num_words, hidden_size))  # 共3444个单词

for k in voc.index2word.keys():

    word = voc.index2word[k]
    if word in vectors.keys():
        voc_emb[k] = vectors[word]
    else:
        voc_emb[k] = np.random.normal(scale=0.6, size=(hidden_size, ))

# TODO 不使用随机而是计算未知词的均值
pickle.dump(voc_emb, open(f'data/embedding_bq.pkl', 'wb'))
