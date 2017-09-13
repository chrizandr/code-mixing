"""LSTM for learning aligned representation."""

import numpy as np
import json
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Lambda
from keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed
from keras.models import Model
from keras import backend as K

from data_handler import text_from_json, break_in_subword
import pdb


def maptosubword(files):
    """Create a map of indices for subwords."""
    texts = []

    for f in files:
        text_data = text_from_json(f)
        f_text = break_in_subword(text_data)
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
    """Create a map of indices for words."""
    texts = []

    for f in files:
        text_data = text_from_json(f)
        _, f_text = break_in_subword(text_data, add_word=True)
        for x in f_text:
            texts.append(' '.join(x))

    texts.append(['<<WPAD>>'])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    return tokenizer


def get_labels(f):
    """Get labels for the data."""
    data = json.load(open(f, "r"))
    labels = np.zeros((len(data,), 3))

    for i in range(len(data)):
        if data[i]["sentiment"] == -1:
            labels[i] = [0, 0, 1.0]
        elif data[i]["sentiment"] == 0:
            labels[i] = [0, 1.0, 0]
        elif data[i]["sentiment"] == 1:
            labels[i] = [1.0, 0, 0]
        else:
            print("hell")

    return labels


def split_validation(train_data, train_labels, ratio=0.2):
    """Split data into train and validation sets."""
    indices = np.arange(train_data.shape[0])
    np.random.seed(10)
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    nb_validation_samples = int(ratio * train_data.shape[0])

    x_train = train_data[:-nb_validation_samples]
    y_train = train_labels[:-nb_validation_samples]
    x_val = train_data[-nb_validation_samples:]
    y_val = train_labels[-nb_validation_samples:]

    return x_train, y_train, x_val, y_val


def get_sequences(texts, max_slen, max_wlen, tokenizer):
    """Something."""
    max_sent_len = 0
    max_word_len = 0
    avg_sent_len = 0
    avg_word_len = 0
    n_words = 0
    n_owords = 0
    n_osents = 0

    data = np.full(
        (len(texts), max_slen, max_wlen),
        tokenizer.word_index['<<SPAD>>'],
        dtype='int64')
    for i in range(len(texts)):
        max_sent_len = max(max_sent_len, len(texts[i]))
        avg_sent_len += len(texts[i])
        if len(texts[i]) > max_slen:
            n_osents += 1
        for j in range(len(texts[i])):
            if j < max_slen:
                max_word_len = max(max_word_len, len(texts[i][j]))
                avg_word_len += len(texts[i][j])
                n_words += 1
                if len(texts[i][j]) > max_wlen:
                    n_owords += 1
                for k in range(len(texts[i][j])):
                    if k < MAX_WORD_LENGTH:
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

TRAIN_DATA_FILE = "data/train_data.json"
TEST_DATA_FILE = "data/test_data.json"

MAX_WORD_LENGTH = 5
MAX_SENT_LENGTH = 60
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

################################################################################
print("C1")
tokenizer = maptosubword([TRAIN_DATA_FILE, TEST_DATA_FILE])
print("Map created")

train_texts = break_in_subword(text_from_json(TRAIN_DATA_FILE))
test_texts = break_in_subword(text_from_json(TEST_DATA_FILE))

train_data = get_sequences(train_texts, MAX_SENT_LENGTH, MAX_WORD_LENGTH, tokenizer)
test_data = get_sequences(test_texts, MAX_SENT_LENGTH, MAX_WORD_LENGTH, tokenizer)
pdb.set_trace()
train_labels = get_labels(TRAIN_DATA_FILE)
test_labels = get_labels(TEST_DATA_FILE)

print("Sequence created")

x_train, y_train, x_val, y_val = split_validation(train_data, train_labels)

del train_data
del train_labels

print('Number of positive and negative tweets in training and validation set')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))

################################################################################
################################################################################
# Hierachical Model graph
################################################################################
# Hindi Word2Vec
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
# preds = Dense(3, activation='softmax')(l_lstm_sent)

# https://keras.io/getting-started/functional-api-guide/#shared-layers [Check this link]

# English Word2Vec
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
# preds = Dense(3, activation='softmax')(l_lstm_sent)

loss_layer = Lambda(lambda x, y: K.categorical_crossentropy(x, y), output_shape=(1,))(l_lstm_sent, l_lstm_sent1)

model = Model(inputs=[sentence_input, sentence_input1], outputs=loss_layer)

model.compile(loss='binary_crossentropy',
              optimizer='adamax',
              metrics=['acc'])

print("model fitting - Hierachical LSTM")
print(model.summary())
model.fit([h_train, e_train], y_train, validation_data=([h_val, e_val], y_val),
          nb_epoch=20, batch_size=32)

print("Evaluating model ...")
score, accuracy = model.evaluate(test_data, test_labels, batch_size=32)
print(score, accuracy)
