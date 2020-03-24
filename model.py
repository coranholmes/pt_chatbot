# -*- coding: utf-8 -*-
"""
@author: Chen Weiling
@software: PyCharm
@file: model.py
@time: 3/18/2020 4:52 PM
@comments: 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import jieba
import pickle
from config import *


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", Unk_token: "UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 删除低于特定计数阈值的单词
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", Unk_token: "UNK"}
        self.num_words = 4  # Count default tokens

        for word in keep_words:
            self.addWord(word)


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s, lang="en"):
    if lang == "en":
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
    else:
        s = s.lower().strip()
    return s


# 初始化Voc对象 和 格式化pairs对话存放到list中
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# 如果对 'p' 中的两个句子都低于 MAX_LENGTH 阈值，则返回True
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


# 过滤满足条件的 pairs 对话
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# 使用上面定义的函数，返回一个填充的voc对象和对列表
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


def indexesFromSentence(voc, sentence):
    res = []
    for word in sentence.split(' '):
        if word in voc.word2index.keys():
            res.append(voc.word2index[word])
        else:
            res.append(Unk_token)
    res.append(EOS_token)
    return res


# zip 对数据进行合并了，相当于行列转置了
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


# 记录 PAD_token的位置为0， 其他的为1
def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        # 如果只有一层，那么不进行Dropout，否则使用传入的参数dropout进行GRU的Dropout。
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # outputs：GRU最后一个隐藏层的输出特征(双向输出之和）; shape = (max_length，batch_size，hidden_size）
        # hidden：从GRU更新隐藏状态; shape =(n_layers x num_directions，batch_size，hidden_size）
        return outputs, hidden


# Luong attention layer
# 参考：http://fancyerii.github.io/2019/02/14/chatbot/
# 参考：https://zhuanlan.zhihu.com/p/47063917
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:  # 点乘，加权点乘，加和
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):  # 并非矩阵乘法，而是补足位数后对应未知进行相乘
        # 输入hidden的shape是(1, batch_size=64, hidden_size=500)
        # encoder_outputs的shape是(input_lengths=10, batch_size=64, hidden_size=500)
        # hidden * encoder_output得到的shape是(10, 64, 500)，然后对第3维求和就可以计算出score，shape变为(10,64)
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)  # 计算W
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    # 输入是上一个时刻的隐状态hidden和所有时刻的Encoder的输出encoder_outputs
    # 输出是注意力的概率，也就是长度为input_lengths的向量，它的和加起来是1。
    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        # 把attn_energies从(max_length=10, batch=64)转置成(64, 10)
        attn_energies = attn_energies.t()

        # 使用softmax函数把score变成概率，shape仍然是(64, 10)，然后用unsqueeze(1)变成(64, 1, 10)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # 注意：decoder每一步只能处理一个时刻的数据，因为t时刻计算完了才能计算t+1时刻。
        # input_step的shape是(1, 64)，64是batch，1是当前输入的词ID(来自上一个时刻的输出)
        # 通过embedding层变成(1, 64, 500)，然后进行dropout，shape不变。
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # 把embedded传入GRU进行forward计算
        # 得到rnn_output的shape是(1, 64, 500)
        # hidden是(2, 64, 500)，因为是两层的GRU，所以第一维是2。
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # 计算注意力权重，根据前面的分析，attn_weights的shape是(64, 1, 10)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # encoder_outputs是(10, 64, 500)
        # encoder_outputs.transpose(0, 1)后的shape是(64, 10, 500)
        # bmm是批量的矩阵乘法，第一维是batch，我们可以把attn_weights看成64个(1,10)的矩阵
        # 把encoder_outputs.transpose(0, 1)看成64个(10, 500)的矩阵
        # 那么bmm就是64个(1, 10)矩阵 x (10, 500)矩阵，最终得到(64, 1, 500)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # 把context向量和GRU的输出拼接起来
        # rnn_output从(1, 64, 500)变成(64, 500)
        rnn_output = rnn_output.squeeze(0)
        # context从(64, 1, 500)变成(64, 500)
        context = context.squeeze(1)
        # 拼接得到(64, 1000)
        concat_input = torch.cat((rnn_output, context), 1)
        # self.concat是一个矩阵(1000, 500)，
        # self.concat(concat_input)的输出是(64, 500)
        # 然后用tanh把输出返回变成(-1,1)，concat_output的shape是(64, 500)
        concat_output = torch.tanh(self.concat(concat_input))
        # out是(64, 词典大小voc.num_words)
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


def maskNLLLoss(inp, target, mask):
    # 计算实际的词的个数，因为padding是0，非padding是1，因此sum就可以得到词的个数
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: 下一个时刻的输入是当前正确答案
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])  # nTotal指这一个batch有几个有效词
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 不是teacher forcing: 下一个时刻的输入是当前模型预测概率最高的值（topi为一个存有index的list）
            _, topi = decoder_output.topk(1)  # decoder_output是当前时刻输出每个词的概率，shape是(batch_size, voc.num_words)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])  # topk的shape是(batch_size, 1)
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    # loss为nTotal个词平均后的loss
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding,
               encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip,
               corpus_name, loadFilename):
    # Load batches for each iteration
    # 随机选择n_iteration个batch的数据(pair)
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
                                                                                          iteration / n_iteration * 100,
                                                                                          print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if iteration % save_every == 0:
            directory = os.path.join(save_dir, model_name, corpus_name,
                                     '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # TODO: NEEDS CHECK HERE!!!
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # decoder_outputs是(batch=1, vob_size)
            # 使用max返回概率最大的词和得分
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # decoder_input是当前时刻输出的词的ID，这是个一维的向量，因为max会减少一维。
            # 但是decoder要求有一个batch维度，因此用unsqueeze增加batch维度。
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def postProcessOutput(text):
    if debug_gen:
        separator = " "
        res = separator.join(text)
        print("output length %d" % len(text))
        print('Debug: ', res)
    # else:
    separator = ""
    res = separator.join(text)
    res = res.replace("?", "？")
    pos = res.find("。")
    if pos > 0:
        res = res[:pos + 1]
    pos = res.find("？")
    if pos > 0:
        res = res[:pos + 1]
    return res


def evaluateInput(searcher, voc):
    input_sentence = ''
    while 1:
        # Get input sentence
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        res = generateAnswer(input_sentence, searcher, voc)
        if lang == "cn":
            print('学长: ', res)
        else:
            print('Bot:', res)


def generateAnswer(query, searcher, voc):
    # Get input sentence
    input_sentence = query
    # Normalize sentence
    input_sentence = normalizeString(input_sentence, lang)
    if lang == "cn":  # 如果是中文则进行分词
        input_sentence = list(jieba.cut(input_sentence))
        if debug_gen:
            print("input length %d" % len(input_sentence))
        input_sentence = " ".join(input_sentence)
    # Evaluate sentence
    output_words = evaluate(searcher, voc, input_sentence)
    # Format and print response sentence
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    if lang == "cn":
        res = postProcessOutput(output_words)
    else:
        separator = " "
        res = separator.join(output_words)
    return res


def initGenModel():
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
    return encoder, decoder, voc, pairs, embedding
