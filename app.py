# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: app.py.py
@time: 3/23/2020 1:59 PM
@comments: 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
from model import *
import warnings
from flask import Flask, render_template, request, make_response
from flask import jsonify

import sys
import time
import hashlib
import threading

warnings.filterwarnings("ignore")


def heartbeat():
    print(time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()


timer = threading.Timer(60, heartbeat)
timer.start()

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import re

zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

app = Flask(__name__, static_url_path="/static")


@app.route('/message', methods=['POST'])
def reply():
    req_msg = request.form['msg']
    print('Message received:', req_msg)
    res_msg = generateAnswer(req_msg, encoder, decoder, searcher, voc)
    print('Message sent:', res_msg)
    # 如果接受到的内容为空，则给出相应的回复
    if res_msg == '':
        res_msg = '你好，我现在有事不在，一会再和你联系。'
    return jsonify({'text': res_msg})


@app.route("/")
def index():
    return render_template("index.html")


with open(vocFile, 'rb') as f:
    voc = pickle.load(f)
with open(pairsFile, 'rb') as f:
    pairs = pickle.load(f)

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)

# 载入预训练的词向量
if loadFilename:
    embedding.load_state_dict(embedding_sd)
elif embeddingFile:
    with open(embeddingFile, 'rb') as f:
        emb = pickle.load(f)
    emb = torch.from_numpy(emb)  # 不转换会报错TypeError: 'int' object is not callable
    embedding.load_state_dict({'weight': emb})

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# 结巴分词准备
init = "".join(list(jieba.cut("聊天系统初始化成功")))

# 启动APP
app.run()

# while 1:
#     # Get input sentence
#     input_sentence = input('> ')
#     # Check if it is quit case
#     if input_sentence == 'q' or input_sentence == 'quit': break
#     ans = generateAnswer(input_sentence, encoder, decoder, searcher, voc)
#     print("学长: ", ans)