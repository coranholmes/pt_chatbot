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


debug = True
init = "".join(list(jieba.cut("聊天系统初始化成功")))
sent_emb = pd.read_pickle(sentEmbFile)
while 1:
    # Get input sentence
    input_sentence = input('> ')
    # Check if it is quit case
    if input_sentence == 'q' or input_sentence == 'quit':
        break
    res, sim = retrieveAnswer(input_sentence, sent_emb)
    print('学长:', res)
