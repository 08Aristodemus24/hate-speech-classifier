import tensorflow as tf
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy as cce_metric
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.optimizers import Adam

import json



def baseline_model():

    model = Sequential()
    model.add(LSTM(64))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='linear'))

    model.compile(
        loss=cce_loss(from_logits=True),
        optimizer=Adam(learning_rate=0.001),
        metrics=[cce_metric(from_logits=True), CategoricalAccuracy()]
    )

    return model