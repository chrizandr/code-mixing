import numpy as np
import pandas as pd
from collections import defaultdict
import re
import json

from bs4 import BeautifulSoup

import sys
import os

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Lambda
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from ortho import ortho_syllable
################################################################################
################################################################################

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

def break_in_subword(f, add_word = False):

    texts = []
    word_texts = []

    data = json.load(open(f,"r"))
    for x in data:
        text = x["text"]
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        cleaned_text = clean_str(text)
        splitted_text = cleaned_text.split()
        joined_text = []
        word_list = []
        for y in splitted_text:
            if(not y.isspace()):
                joined_text.append(ortho_syllable(y.strip()))
                if add_word:
                    word_list.append(y.strip())
        texts.append(joined_text)
        if add_word:
            word_texts.append(word_list)

    if add_word:
        return texts, word_texts
    else:
        return texts


def maptosubword(files):

    texts = []

    for f in files:
        f_text = break_in_subword(f)
        for x in f_text:
            temp = []
            for y in x:
                temp.extend(y)
            texts.append(str(' '.join(temp)))

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(texts)

    tokenizer.word_index['<<SPAD>>'] = len(tokenizer.word_index) + 1

    return tokenizer

def maptoword(files):

    texts = []

    for f in files:
        _, f_text = break_in_subword(f, add_word=True)
        for x in f_text:
            texts.append(' '.join(x))

    texts.append(['<<WPAD>>'])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    return tokenizer

def get_labels(f):

    data = json.load(open(f,"r"))
    labels = np.zeros((len(data,), 3))

    for i in range(len(data)):
        if data[i]["sentiment"] == -1:
            labels[i] = [0,0,1.0]
        elif data[i]["sentiment"] == 0:
            labels[i] = [0,1.0,0]
        elif data[i]["sentiment"] == 1:
            labels[i] = [1.0,0,0]
        else:
            print("hell")

    return labels

def get_sequences(texts, max_slen, max_wlen, tokenizer):

    max_sent_len = 0
    max_word_len = 0
    avg_sent_len = 0
    avg_word_len = 0
    n_words = 0
    n_owords = 0
    n_osents = 0

    data = np.full(
        (len(texts), max_slen, max_wlen),
        tokenizer.word_index['<<SPAD>>'], dtype='int64')

    for i in range(len(texts)):
        max_sent_len = max(max_sent_len, len(texts[i]))
        avg_sent_len += len(texts[i])
        if len(texts[i]) > max_slen:
            n_osents += 1
        for j in range(len(texts[i])):
            if j<max_slen:
                max_word_len = max(max_word_len, len(texts[i][j]))
                avg_word_len += len(texts[i][j])
                n_words += 1
                if len(texts[i][j])> max_wlen:
                    n_owords += 1
                for k in range(len(texts[i][j])):
                    if k<MAX_WORD_LENGTH:
                        if texts[i][j][k] in tokenizer.word_index:
                            data[i][j][k] = tokenizer.word_index[texts[i][j][k]]
                        else:
                            data[i][j][k] = 0
                    else:
                        break
            else:
                break

    print("Max sent len {}, Max word len {}".format(max_sent_len, max_word_len))
    print("Avg sent len {}, Abg word len {}".format(avg_sent_len/len(texts), avg_word_len/n_words))
    print("Out sent {}, Out words {}".format(n_osents, n_owords))

    return data

################################################################################
################################################################################

TRAIN_DATA_FILE = "../Data/CODEMIXED_TRAIN_DATA.json"
TEST_DATA_FILE = "../Data/CODEMIXED_TEST_DATA.json"

MAX_WORD_LENGTH = 5
MAX_SENT_LENGTH = 60
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

################################################################################
################################################################################
tokenizer = maptosubword([TRAIN_DATA_FILE, TEST_DATA_FILE])
print("Map created")

train_texts = break_in_subword(TRAIN_DATA_FILE)
train_labels = get_labels(TRAIN_DATA_FILE)
test_texts = break_in_subword(TEST_DATA_FILE)
test_labels = get_labels(TEST_DATA_FILE)

train_data = get_sequences(train_texts, MAX_SENT_LENGTH, MAX_WORD_LENGTH, tokenizer)
test_data = get_sequences(test_texts, MAX_SENT_LENGTH, MAX_WORD_LENGTH, tokenizer)
print("Sequence created")

indices = np.arange(train_data.shape[0])
np.random.seed(10)
np.random.shuffle(indices)
train_data = train_data[indices]
train_labels = train_labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * train_data.shape[0])

x_train = train_data[:-nb_validation_samples]
y_train = train_labels[:-nb_validation_samples]
x_val = train_data[-nb_validation_samples:]
y_val = train_labels[-nb_validation_samples:]

del train_data
del train_labels

print('Number of positive and negative tweets in training and validation set')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))

################################################################################
################################################################################
# Hierachical Model graph
################################################################################
##Hindi Word2Vec
embedding_matrix = np.random.uniform(
    low=0.1, high=0.5,
    size=(len(tokenizer.word_index)+1, EMBEDDING_DIM))

embedding_layer = Embedding(len(tokenizer.word_index)+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_WORD_LENGTH,
                            trainable=True)

word_input = Input(shape=(MAX_WORD_LENGTH,), dtype='int64')
embedded_sequences = embedding_layer(word_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
word_encoder = Model(word_input, l_lstm)

sentence_input = Input(shape=(MAX_SENT_LENGTH, MAX_WORD_LENGTH), dtype='int64')
sentence_encoder = TimeDistributed(word_encoder)(sentence_input)
l_lstm_sent = Bidirectional(LSTM(100))(sentence_encoder)
#preds = Dense(3, activation='softmax')(l_lstm_sent)


#English Word2Vec
embedding_matrix1 = np.random.uniform(
    low=0.1, high=0.5,
    size=(len(tokenizer.word_index)+1, EMBEDDING_DIM))

embedding_layer1 = Embedding(len(tokenizer.word_index)+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix1],
                            input_length=MAX_WORD_LENGTH,
                            trainable=True)

word_input1 = Input(shape=(MAX_WORD_LENGTH,), dtype='int64')
embedded_sequences1 = embedding_layer(word_input1)
l_lstm1 = Bidirectional(LSTM(100))(embedded_sequences1)
word_encoder1 = Model(word_input1, l_lstm1)

sentence_input1 = Input(shape=(MAX_SENT_LENGTH, MAX_WORD_LENGTH), dtype='int64')
sentence_encoder1 = TimeDistributed(word_encoder1)(sentence_input1)
l_lstm_sent1 = Bidirectional(LSTM(100))(sentence_encoder1)
#preds = Dense(3, activation='softmax')(l_lstm_sent)

loss_layer = Lambda(lambda x,y: K.categorical_crossentropy(x, y), output_shape=(1,))(l_lstm_sent, l_lstm_sent1)

model = Model(inputs=[sentence_input,sentence_input1], outputs=loss_layer)

model.compile(loss='binary_crossentropy',
              optimizer='adamax',
              metrics=['acc'])

print("model fitting - Hierachical LSTM")
print(model.summary())
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=20, batch_size=32)

print("Evaluating model ...")
score, accuracy = model.evaluate(test_data, test_labels, batch_size=32)
print(score, accuracy)


