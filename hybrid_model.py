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
import warnings, sys

warnings.filterwarnings("ignore")

start = time.time()
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

encoder, decoder, voc, pairs, embedding = initGenModel()
encoder.eval()
decoder.eval()
searcher = GreedySearchDecoder(encoder, decoder)

end = time.time()
print("Finish loading the environment: ", end - start, "s")

while 1:
    input_sentence = input('> ')
    retrieve_win = True
    if input_sentence == 'q' or input_sentence == 'quit':
        break
    if retrieve_mode == "brute_force":
        res_ret, sim = retrieveAnswer(input_sentence, sent_emb)
        if sim < threshold_ret:
            retrieve_win = False
    elif retrieve_mode == "annoy":
        res_ret, sim = retrieveAnswer_ann(input_sentence, sent_emb, u)
        if sim > threshold_ann:
            retrieve_win = False
    elif retrieve_mode == "ball_tree":
        res_ret, sim = retrieveAnswer_bt(input_sentence, sent_emb, bt)
        if sim > threshold_tree:
            retrieve_win = False
    else:
        print("Wrong retrieve mode!!!")
        sys.exit(0)
    res_gen = generateAnswer(input_sentence, searcher, voc)
    if not retrieve_win:
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
