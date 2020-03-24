# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: retrieval.py
@time: 3/24/2020 10:06 AM
@comments: 测试检索模型
"""
from utils import *
import jieba
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def computeSimilarity(row, emb):
    emb2 = row['emb'].reshape(1, -1)
    sim = cosine_similarity(emb, emb2)
    return sim.item()


debug = True
init = "".join(list(jieba.cut("聊天系统初始化成功")))
sent_emb = pd.read_pickle('data/sent_emb.pkl')
while 1:
    # Get input sentence
    input_sentence = input('> ')
    # Check if it is quit case
    if input_sentence == 'q' or input_sentence == 'quit': break
    input_sentence = " ".join(list(jieba.cut(input_sentence)))
    emb = computeSentEmb(input_sentence)
    emb = emb.reshape(1, -1)
    sim = sent_emb.apply(computeSimilarity, axis=1, args=(emb,))
    sent_emb['sim'] = sim
    # print(sent_emb.nlargest(10, 'sim'))
    max_sim = sent_emb['sim'].max()
    res = sent_emb[sent_emb['sim'] == max_sim]
    if max_sim < 0.9965:
        print("未检索到结果！降低阈值查找中...")
        if debug:
            res = sent_emb.nlargest(10, 'sim')
            print(res.loc[:, ['qry', 'ans', 'sim']])
    res = res.sample(1).ans.values.item()  # 转换为str格式
    res = res.replace(" ", "")
    print('学长:', res)
