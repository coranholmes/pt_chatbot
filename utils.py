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
from annoy import AnnoyIndex
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

    if debug_ret:
        if max_sim < threshold_ret:
            print("阈值%.4f未检索到结果！搜索到的最大相似度为%f。降低阈值查找中..." % (threshold_ret, max_sim))
        else:
            print("检索到结果！")
        top10 = sent_emb.nlargest(10, 'sim')
        print(top10.loc[:, ['qry', 'ans', 'sim']])

    return res, max_sim


def retrieveAnswer_bt(query, sent_emb, kdt):
    input_sentence = normalizeString(query, lang)
    input_sentence = " ".join(list(jieba.cut(input_sentence)))
    emb = computeSentEmb(input_sentence)
    emb = emb.reshape(1, -1)
    dist, ind = kdt.query(emb, k=10)
    min_sim = dist.min()
    cand_idx = np.argwhere(dist == min_sim)
    res_idx = ind[0][np.random.choice(cand_idx[:, 1])]
    res = sent_emb.loc[res_idx, 'ans']
    res = res.replace(" ", "")

    if debug_ret:
        if min_sim > threshold_tree:
            print("阈值%.4f未检索到结果！搜索到的最短距离为%f。升高阈值查找中..." % (threshold_tree, min_sim))
        else:
            print("检索到结果！")
        for _, i in enumerate(ind[0]):
            print(sent_emb.loc[i, 'qry'], "\t", sent_emb.loc[i, 'ans'], "\t", dist[0][_])
    return res, min_sim


def retrieveAnswer_ann(query, sent_emb, u):
    input_sentence = normalizeString(query, lang)
    input_sentence = " ".join(list(jieba.cut(input_sentence)))
    emb = computeSentEmb(input_sentence)
    ind, dist = u.get_nns_by_vector(emb, 10, search_k=19000, include_distances=True)

    ind = np.array(ind)
    dist = np.array(dist)
    min_sim = dist.min()
    cand_idx = np.argwhere(dist == min_sim)
    res_idx = ind[np.random.choice(cand_idx.ravel())]
    res = sent_emb.loc[res_idx, 'ans']
    res = res.replace(" ", "")

    if debug_ret:
        if min_sim > threshold_ann:
            print("阈值%.4f未检索到结果！搜索到的最短距离为%f。升高阈值查找中..." % (threshold_tree, min_sim))
        else:
            print("检索到结果！")
        for _, i in enumerate(ind):
            print(sent_emb.loc[i, 'qry'], "\t", sent_emb.loc[i, 'ans'], "\t", dist[_])
    return res, min_sim
