# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: compute_sent_emb.py
@time: 3/23/2020 11:10 PM
@comments: 对知识库现有的query计算sentence embedding备用
"""


from utils import *

# 因为通过gensim来load fasttext预训练的词向量速度非常慢，所以这里先load一次之后保存成pickle，之后读取pickle速度会快一些
# model = KeyedVectors.load_word2vec_format(fastTextEmb)
# with open(fastTextGensim, 'wb') as f:
#     pickle.dump(model, f)


def applyfn(row):
    query = row['qry']
    return computeSentEmb(query)


# print(model.most_similar('老师'))
# print(model.similarity('老师', '学生'))
# print(model.wv['老师'])

data = pd.read_table(dialogFile, sep="\t", header=None, index_col=None)
data.columns = ['qry', 'ans']
# with open('data/stop_words.txt', encoding="utf-8") as f:  # 闲聊废话比较多，加了停用词过滤很多话没有有效的词
#     stopWords = [line.strip() for line in f.readlines()]
data['emb'] = data.apply(applyfn, axis=1)
data.to_pickle('data/sent_emb.pkl')
