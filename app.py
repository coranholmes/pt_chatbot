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
    res_msg = generateAnswer(req_msg, searcher, voc)
    print('Message sent:', res_msg)
    # 如果接受到的内容为空，则给出相应的回复
    if res_msg == '':
        res_msg = '你好，我现在有事不在，一会再和你联系。'
    return jsonify({'text': res_msg})


@app.route("/")
def index():
    return render_template("index.html")


encoder, decoder, voc, pairs, embedding = initGenModel()

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# 结巴分词准备
init = "".join(list(jieba.cut("聊天系统初始化成功")))

# 启动APP
app.run()
