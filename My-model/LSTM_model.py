#!~/anaconda3/bin/python3.6
# encoding: utf-8

"""
@author: Yongbo Wang
@file: ToxicClassification - Bi-LSTM_model.py
@time: 9/1/18 4:47 AM
"""
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

DATA_PATH = '../input'
EMBEDDING_FILE = f'{DATA_PATH}/glove6b50d/glove.6B.50d.txt'
TRAIN_DATA_FILE = f'{DATA_PATH}/train.csv'
TEST_DATA_FILE = f'{DATA_PATH}/test.csv'
SAVE_FILE_PATH = 'lstm.best.hdf5'


class BiLSTMModel:
    def __init__(self):
        # the size of each word vector
        self.embed_size = 50
        self.max_features = 20000
        self.maxlen = 100

        self.train = pd.read_csv(TRAIN_DATA_FILE)
        self.test = pd.read_csv(TEST_DATA_FILE)
        self.list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.tokenizer = Tokenizer(num_words=self.max_features)

        self.learning_rate = 0.001
        self.batch_size = 10
        self.epochs = 2
        self.dropout_rate = 0.1

        self.X_t = np.array([])
        self.X_te = np.array([])
        self.y = np.array([])

    def preprocess(self):
        list_sentences_train = self.train["comment_text"].fillna("_na_").values
        self.y = self.train[self.list_classes].values
        list_sentences_test = self.test["comment_text"].fillna("_na_").values

        self.tokenizer.fit_on_texts(list(list_sentences_train))
        list_tokenized_train = self.tokenizer.texts_to_sequences(list_sentences_train)
        list_tokenized_test = self.tokenizer.texts_to_sequences(list_sentences_test)
        self.X_t = pad_sequences(list_tokenized_train, maxlen=self.maxlen)
        self.X_te = pad_sequences(list_tokenized_test, maxlen=self.maxlen)

    def get_trans(self, word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def word_embedding(self):
        # Read the glove word vectors into a dictionary
        embeddings_index = dict(self.get_trans(*o.strip().split()) for o in open(EMBEDDING_FILE))

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()

        word_index = self.tokenizer.word_index
        nb_words = min(self.max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, self.embed_size))
        for word, i in word_index.items():
            if i >= self.max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def get_model(self):
        embedding_matrix = self.word_embedding()
        x_input = Input(shape=(self.maxlen,))
        x = Embedding(self.max_features, self.embed_size, weights=[embedding_matrix])(x_input)
        x = LSTM(50, dropout=0.1, recurrent_dropout=0.1)(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=x_input, outputs=x)
        optimizer = optimizers.Adam(lr=self.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # early stop
        checkpoint = ModelCheckpoint(SAVE_FILE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
        callbacks_list = [checkpoint, early]
        model.fit(self.X_t, self.y,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_split=0.1,
                  callbacks=callbacks_list)

        return model

    def test_model(self):
        model = self.get_model()
        y_test = model.predict([self.X_te], batch_size=1024, verbose=1)
        # model.evaluate()
        sample_submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')
        sample_submission[self.list_classes] = y_test
        sample_submission.to_csv('lstm_submission.csv', index=False)


if __name__ == '__main__':
    blm = BiLSTMModel()
    blm.preprocess()
    blm.test_model()
