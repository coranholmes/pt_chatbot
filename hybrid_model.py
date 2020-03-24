# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: hybrid_model.py
@time: 3/24/2020 1:34 PM
@comments: 生成+检索混合模型
"""

from config import *
from model import *
from utils import *
import warnings

warnings.filterwarnings("ignore")

start = time.time()
init = "".join(list(jieba.cut("聊天系统初始化成功")))

sent_emb = pd.read_pickle(sentEmbFile)

encoder, decoder, voc, pairs, embedding = initGenModel()
encoder.eval()
decoder.eval()
searcher = GreedySearchDecoder(encoder, decoder)

end = time.time()
print("Finish loading the environment: ", end - start, "s")

while 1:
    input_sentence = input('> ')
    # Check if it is quit case
    if input_sentence == 'q' or input_sentence == 'quit':
        break
    res_ret, sim = retrieveAnswer(input_sentence, sent_emb)
    res_gen = generateAnswer(input_sentence, searcher, voc)
    if sim < threshold_ret:
        print("!!!采用【生成模型】结果!!!")
        res = res_gen
    else:
        print("!!!采用【检索模型】结果!!!")
        res = res_ret
    if debug_hyb:
        print("生成:", res_gen)
        print("检索:", res_ret, sim)
    else:
        print("学长:", res)
