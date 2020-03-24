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
from model import *
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors


with open(fastTextGensim, 'rb') as f:  # 直接load_word2vec_format太慢了，所以读取之后保存成pickle再读
    model = pickle.load(f)


def computeSentEmb(sent):
    words = sent.split()
    res = np.zeros((hidden_size,), dtype=np.float32)
    # words = [w for w in words if w not in stopWords]
    for w in words:
        if w in model.wv:
            res = res + model.wv[w]
        else:
            # print(w)
            res = res + np.random.normal(scale=0.3, size=(hidden_size,))
    res = res / hidden_size
    return res


def computeSimilarity(row, emb):
    emb2 = row['emb'].reshape(1, -1)
    sim = cosine_similarity(emb, emb2)
    return sim.item()


def retrieveAnswer(query, sent_emb):
    input_sentence = normalizeString(query, lang)
    input_sentence = " ".join(list(jieba.cut(input_sentence)))
    emb = computeSentEmb(input_sentence)
    emb = emb.reshape(1, -1)
    sim = sent_emb.apply(computeSimilarity, axis=1, args=(emb,))
    sent_emb['sim'] = sim
    # print(sent_emb.nlargest(10, 'sim'))
    max_sim = sent_emb['sim'].max()
    res = sent_emb[sent_emb['sim'] == max_sim]
    res = res.sample(1).ans.values.item()  # 转换为str格式
    res = res.replace(" ", "")

    if max_sim < threshold_ret:
        if debug_ret:
            print("阈值%.4f未检索到结果！搜索到的最大相似度为%f。降低阈值查找中..." % (threshold_ret, max_sim))
            top10 = sent_emb.nlargest(10, 'sim')
            print(top10.loc[:, ['qry', 'ans', 'sim']])

    return res, max_sim
