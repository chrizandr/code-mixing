"""LSTM for learning aligned representation."""

import numpy as np
import json
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Lambda
from keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed
from keras.models import Model
from keras.losses import mean_squared_error
from keras.layers.merge import Concatenate
from data_handler import text_from_json, break_in_subword, read_data
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


def get_embeddings_tokenizer(filename, EMBEDDING_DIM):
    """Get the embeddings and the tokenizer for words."""
    data = read_data(filename)
    texts = []

    embedding_matrix = np.random.randn(len(data), EMBEDDING_DIM)
    embedding_matrix = embedding_matrix.astype(np.float64)
    for i in range(len(data)):
        raw = data[i].split()
        label = raw[0]
        vector = [float(x) for x in raw[1:EMBEDDING_DIM+1]]
        texts.append(label)
        embedding_matrix[i] = vector

    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(texts)
    word_tokenizer.word_index['<<SPAD>>'] = len(word_tokenizer.word_index) + 1

    return embedding_matrix, word_tokenizer


def create_model(EMBEDDING_DIM, MAX_WORD_LENGTH, MAX_SENT_LENGTH):
    """Create model for alignment of word embeddings."""
    word_lstm = Bidirectional(LSTM(EMBEDDING_DIM))
    sentence_lstm = Bidirectional(LSTM(EMBEDDING_DIM))
    # -------------------------------------------------------
    # Hindi Pass
    hindi_word_syllables = Input(shape=(MAX_WORD_LENGTH,), dtype='int64')
    hindi_sentence_words = Input(shape=(MAX_SENT_LENGTH,), dtype='int64')
    hindi_sentence_word_syll = Input(shape=(MAX_SENT_LENGTH, MAX_WORD_LENGTH), dtype='int64')

    hindi_word_embedding_layer = Embedding(len(hindi_word_tokenizer.word_index),
                                           EMBEDDING_DIM,
                                           weights=[hindi_embedding_matrix],
                                           input_length=EMBEDDING_DIM,
                                           trainable=True)

    # Word model
    hindi_word_lstm = word_lstm(hindi_word_syllables)
    hindi_word_model = Model(hindi_word_syllables, hindi_word_lstm)

    hindi_word_embeddings = hindi_word_embedding_layer(hindi_sentence_words)
    hindi_sentence = TimeDistributed(hindi_word_model)(hindi_sentence_word_syll)

    # Sentence Model
    hindi_sentence_representation = Concatenate([hindi_word_embeddings, hindi_sentence])
    hindi_sentence_lstm = sentence_lstm(hindi_sentence_representation)

    # -------------------------------------------------------
    # Egnlish Pass

    eng_word_syllables = Input(shape=(MAX_WORD_LENGTH,), dtype='int64')
    eng_sentence_words = Input(shape=(MAX_SENT_LENGTH,), dtype='int64')
    eng_sentence_word_syll = Input(shape=(MAX_SENT_LENGTH, MAX_WORD_LENGTH), dtype='int64')

    eng_word_embedding_layer = Embedding(len(eng_word_tokenizer.word_index),
                                         EMBEDDING_DIM,
                                         weights=[eng_embedding_matrix],
                                         input_length=EMBEDDING_DIM,
                                         trainable=True)

    # Word model
    eng_word_lstm = word_lstm(eng_word_syllables)
    eng_word_model = Model(eng_word_syllables, eng_word_lstm)

    eng_word_embeddings = eng_word_embedding_layer(eng_sentence_words)
    eng_sentence = TimeDistributed(eng_word_model)(eng_sentence_word_syll)

    # Sentence Model
    eng_sentence_representation = Concatenate([eng_word_embeddings, eng_sentence])
    eng_sentence_lstm = sentence_lstm(eng_sentence_representation)

    loss_layer = Lambda(lambda x: mean_squared_error(x[0], x[1]), output_shape=(1,))([eng_sentence_lstm, hindi_sentence_lstm])

    model = Model(inputs=[eng_sentence_words, eng_sentence_word_syll, hindi_sentence_words, hindi_sentence_word_syll], outputs=loss_layer)

    return model


if __name__ == "__main__":
    TRAIN_DATA_FILE = "data/train_data.json"
    TEST_DATA_FILE = "data/test_data.json"

    MAX_WORD_LENGTH = 5
    MAX_SENT_LENGTH = 20
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    hindi_embedding_matrix, hindi_word_tokenizer = get_embeddings_tokenizer("data/parallel.hi", EMBEDDING_DIM)
    eng_embedding_matrix, eng_word_tokenizer = get_embeddings_tokenizer("data/parallel.en", EMBEDDING_DIM)

    _, hindi_syllable_tokenize = get_embeddings_tokenizer("data/parallel.hi.syll", EMBEDDING_DIM)
    _, eng_syllable_tokenizer = get_embeddings_tokenizer("data/parallel.en.syll", EMBEDDING_DIM)

    model = create_model(EMBEDDING_DIM, MAX_WORD_LENGTH, MAX_SENT_LENGTH)
    pdb.set_trace()
    model.compile(loss='mean_squared_error',
                  optimizer='adamax',
                  metrics=['acc'])

    # print("model fitting - Hierachical LSTM")
    # print(model.summary())
    # model.fit([h_train, e_train], y_train, validation_data=([h_val, e_val], y_val),
    #           nb_epoch=20, batch_size=32)
    #
    # print("Evaluating model ...")
    # score, accuracy = model.evaluate(test_data, test_labels, batch_size=32)
    # print(score, accuracy)
