# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: generate_voc_pairs.py
@time: 3/19/2020 2:07 PM
@comments: 生成voc和pairs
"""

import os, re, pickle, random
import jieba
import pandas as pd
from model import Voc, filterPairs, trimRareWords, batch2TrainData
from config import *


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    return s


# 初始化Voc对象 和 格式化pairs对话存放到list中
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    pairs = []
    with open(datafile, 'r', encoding="utf-8") as f:
        for lines in f:
            parts = lines.strip().split("\t")
            if len(parts) != 2:
                continue
            pairs.append([parts[0], parts[1]])
    voc = Voc(corpus_name)
    return voc, pairs

def loadPrepareData(path, corpus_name):
    print("Start preparing training data ...")
    voc, pairs = readVocs(path, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


if __name__ == '__main__':
    # Load/Assemble voc and pairs
    voc, pairs = loadPrepareData(dialogFile, 'baiqi')
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    pairs = trimRareWords(voc, pairs, 2)  # MIN_COUNT = 2

    print(len(voc.word2index))
    print(len(pairs))

    with open('data/voc_bq.pkl','wb') as f:
        pickle.dump(voc, f)
    with open('data/pairs_bq.pkl','wb') as f:
        pickle.dump(pairs, f)

    # Example for validation
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    with open('data/batches.pkl', 'wb') as f:
        pickle.dump(batches, f)
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)