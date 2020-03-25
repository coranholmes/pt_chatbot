# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: retrieval.py
@time: 3/24/2020 10:06 AM
@comments: 测试检索模型
"""
from utils import *
import jieba, sys
import pandas as pd
import numpy as np


init = "".join(list(jieba.cut("聊天系统初始化成功")))
sent_emb = pd.read_pickle(sentEmbFile)

if retrieve_mode == "brute_force":
    print("已选择Brute force检索...")
elif retrieve_mode == "annoy":
    print("已选择Annoy index检索...")
    u = AnnoyIndex(hidden_size, 'angular')
    u.load(annoyIdxFile)
elif retrieve_mode == "ball_tree":
    print("已选择Ball Tree检索...")
    with open(ballTreeIdxFile, 'rb') as f:
        bt = pickle.load(f)
else:
    print("Wrong retrieve mode!!!")
    sys.exit(0)

while 1:
    input_sentence = input('> ')
    if input_sentence == 'q' or input_sentence == 'quit':
        break
    if retrieve_mode == "brute_force":
        res_ret, sim = retrieveAnswer(input_sentence, sent_emb)
    elif retrieve_mode == "annoy":
        res_ret, sim = retrieveAnswer_ann(input_sentence, sent_emb, u)
    elif retrieve_mode == "ball_tree":
        res_ret, sim = retrieveAnswer_bt(input_sentence, sent_emb, bt)
    else:
        print("Wrong retrieve mode!!!")
        sys.exit(0)
    print('学长:', res_ret)
