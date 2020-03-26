# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: train.py
@time: 3/18/2020 5:12 PM
@comments: 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from model import *
import warnings

warnings.filterwarnings("ignore")

encoder, decoder, voc, pairs, embedding = initGenModel()

if mode == "train":
    print("Training Mode starts ...")
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # If you have cuda, configure cuda to call TODO: NEEDS CHECK HERE!!!
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")
    save_dir = os.path.join("data", "save")
    trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, loadFilename)

print("Evaluation Mode starts ...")
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# 结巴分词准备
init = "".join(list(jieba.cut("聊天系统初始化成功")))

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(searcher, voc)

