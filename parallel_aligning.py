"""LSTM for learning aligned representation."""

import numpy as np
import pdb
import json
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Lambda
from keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed, Dense, Add, Subtract, Dot
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from data_handler import break_in_subword, read_data


def read_layered_subword(filename):
    """Read data as subwords."""
    text_data = read_data(filename)

    text_layered = break_in_subword(text_data)

    return text_layered


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
                    if k < max_wlen:
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


def get_embeddings_tokenizer(filename1, filename2, EMBEDDING_DIM):
    """Get the embeddings and the tokenizer for words."""
    data = read_data(filename1) + read_data(filename2)
    texts = []

    embedding_matrix = np.zeros((len(data)+1, EMBEDDING_DIM))
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


def create_model(EMBEDDING_DIM, MAX_WORD_LENGTH, MAX_SENT_LENGTH, NUM_SUBWORDS, subword_embeddings, alpha):
    """Create model for alignment of word embeddings."""
    word_lstm = Bidirectional(LSTM(EMBEDDING_DIM), name="word_lstm")
    sentence_lstm = Bidirectional(LSTM(EMBEDDING_DIM), name="sentence_lstm")
    # -------------------------------------------------------
    # Hindi Pass
    hindi_sentence = Input(shape=(MAX_SENT_LENGTH, MAX_WORD_LENGTH), dtype='int64')
    eng_sentence = Input(shape=(MAX_SENT_LENGTH, MAX_WORD_LENGTH), dtype='int64')
    # hindi_aligned = Input(shape=(MAX_WORD_LENGTH,), dtype='int64')
    # eng_aligned = Input(shape=(MAX_WORD_LENGTH,), dtype='int64')

    subword_embedding_layer = Embedding(NUM_SUBWORDS,
                                        EMBEDDING_DIM,
                                        weights=[subword_embeddings],
                                        input_length=MAX_WORD_LENGTH,
                                        trainable=True,
                                        name="subword_embedding")

    # Word model
    hindi_word_embeddings = TimeDistributed(subword_embedding_layer)(hindi_sentence)
    hindi_word_out = TimeDistributed(word_lstm)(hindi_word_embeddings)
    hindi_sentence_out = sentence_lstm(hindi_word_out)

    eng_word_embeddings = TimeDistributed(subword_embedding_layer)(eng_sentence)
    eng_word_out = TimeDistributed(word_lstm)(eng_word_embeddings)
    eng_sentence_out = sentence_lstm(eng_word_out)

    sentence_diff = Subtract()([hindi_sentence_out, eng_sentence_out])
    norm_sentence = Dot(axes=1)([sentence_diff, sentence_diff])
    sentence_loss = Lambda(lambda x: x*alpha)(norm_sentence)
    # sentence_loss = Dense(1, input_shape=(1,))(norm_sentence)

    # hindi_aligned_embeddings = subword_embedding_layer(hindi_aligned)
    # eng_aligned_embeddings = subword_embedding_layer(eng_aligned)
    # hindi_aligned_out = word_lstm(hindi_aligned_embeddings)
    # eng_aligned_out = word_lstm(eng_aligned_embeddings)

    # word_diff = Subtract()([hindi_aligned_out, eng_aligned_out])
    # word_square = Multiply()([word_diff, word_diff])
    # norm_word = Lambda(lambda x: K.sum(x), output_shape=lambda s: (s[0], 1))(word_square)
    # word_loss = Dense(1, name="alpha2")(norm_word)

    # total_loss = Add()([word_loss, sentence_loss])

    model = Model(inputs=[hindi_sentence, eng_sentence], outputs=sentence_loss)
    model.compile(loss='mean_squared_error',
                  optimizer='adamax',
                  metrics=['acc'])

    return model


if __name__ == "__main__":
    TRAIN_DATA_FILE = "data/train_data.json"
    TEST_DATA_FILE = "data/test_data.json"
    MODEL_FILE = "parallel_aligning.hd5"

    MAX_WORD_LENGTH = 5
    MAX_SENT_LENGTH = 20
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    checkpoint = ModelCheckpoint(filepath=MODEL_FILE, monitor='val_loss')

    # _, word_tokenizer = get_embeddings_tokenizer("data/parallel.hi", "data/parallel.en", EMBEDDING_DIM)
    subword_embeddings, subword_tokenizer = get_embeddings_tokenizer("data/parallel.hi.syll",
                                                                     "data/parallel.hi.syll",
                                                                     EMBEDDING_DIM)
    layered_data_e = read_layered_subword("data/IITB.en-hi.en")
    layered_data_h = read_layered_subword("data/IITB.en-hi.hi")

    h_sequences = get_sequences(layered_data_h, MAX_SENT_LENGTH, MAX_WORD_LENGTH, subword_tokenizer)
    e_sequences = get_sequences(layered_data_e, MAX_SENT_LENGTH, MAX_WORD_LENGTH, subword_tokenizer)

    split_point = int(h_sequences.shape[0]*VALIDATION_SPLIT)

    hx_train = h_sequences[0:split_point]
    ex_train = e_sequences[0:split_point]
    y_train = np.zeros((hx_train.shape[0], 1))

    hx_val = h_sequences[split_point:]
    ex_val = e_sequences[split_point:]
    y_val = np.zeros((hx_val.shape[0], 1))

    model = create_model(EMBEDDING_DIM, MAX_WORD_LENGTH,
                         MAX_SENT_LENGTH, len(subword_tokenizer.word_index),
                         subword_embeddings,
                         alpha=0.5)

    print("model fitting - Hierachical LSTM")
    print(model.summary())
    model.fit([hx_train, ex_train], y_train, validation_data=([hx_val, ex_val], y_val),
              epochs=20, batch_size=32, callbacks=[checkpoint])
