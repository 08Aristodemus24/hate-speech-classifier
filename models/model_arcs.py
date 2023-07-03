import tensorflow as tf
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Activation, Embedding
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy as cce_metric
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.optimizers import Adam

import json


def init_embedding_layer(vocab_len, emb_matrix):
    emb_dim = emb_matrix.shape[1]
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def load_lstm_model(input_shape, vocab_len, emb_matrix):
    """
    Define architecture of LSTM model then return for later training

    Takes in also the embedding layer with weights/coefficients set
    to the pre-trained GloVe embeddings
    """
    print(f'input shape: {input_shape}')

    seqs_padded = Input(shape=(input_shape, ), dtype='int64')
    embeddings = init_embedding_layer(vocab_len, emb_matrix)(seqs_padded)
    A1 = Bidirectional(LSTM(units=16, return_sequences=True))(embeddings)
    D1 = Dropout(0.5)(A1)
    A2 = Bidirectional(LSTM(units=16, return_sequences=False))(D1)
    D2 = Dropout(0.5)(A2)
    Z3 = Dense(units=4)(D2)
    A3 = Activation('softmax')(Z3)

    model = Model(inputs=seqs_padded, outputs=A3, name='hate-speech-lstm')

    model.compile(
        loss=cce_loss(),
        optimizer=Adam(learning_rate=0.001),
        metrics=[cce_metric(), CategoricalAccuracy()]
    )

    return model



def load_softmax_model():
    pass