import numpy as np
import pandas as pd
import time

import tensorflow as tf 

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
#
class LeNet:
    def __init__(self):
        model = Sequential()
        model.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), padding='same'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2), strides=2))
        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D((2, 2), strides=2))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        self.model = model

    #callback = tf.keras.callbacks.EarlyStopping(monitor ="val_loss", patience = 2)
    def fit(self, X_train, y_train, validation_split =0.2, lr=0.01, epochs=10, batch_size=60, verbose=0):
        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr), metrics=['accuracy'])
        self.result = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, \
                                     validation_split = validation_split, verbose=verbose)

        return self







