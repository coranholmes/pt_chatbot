# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: utils.py
@time: 3/24/2020 11:00 AM
@comments: sentence embedding相关的工具方法
"""


import time, pickle
import pandas as pd
import numpy as np
from config import *
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors


def initFastTextEmb():
    start = time.time()
    with open(fastTextGensim, 'rb') as f:  # 直接load_word2vec_format太慢了，所以读取之后保存成pickle再读
        model = pickle.load(f)
    end = time.time()
    print("Finish loading embedding: ", end - start)
    return model

model = initFastTextEmb()

def computeSentEmb(sent):
    words = sent.split()
    res = np.zeros((hidden_size,), dtype=np.float32)
    # words = [w for w in words if w not in stopWords]
    for w in words:
        if w in model.wv:
            res = res + model.wv[w]
        else:
            # print(w)
            res = res + np.random.normal(scale=0.3, size=(hidden_size, ))
    res = res / hidden_size
    return res
