# IMPORT THE FOLLOWING PACKAGES TO RUN THE CODE CORRECTLY 
#
# MODEL CNN_D : CONVOLUTION NEURAL NETWORK WITH REGULARIZAITON DROPOUT AFTER THE FULLY CONNECTED LAYER 
# X_TRAIN = IMAGES IN THE TRAINING DATA
# Y_TRAIN = LABELS FROM THE TRAINING DATA (0/1)
# VALIDATION_DATA = (X_VALIDATION, Y_VALIDATION)


import numpy as np
import pandas as pd

import tensorflow as tf 

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


class LeNet_MM_D:
    def __init__(self):
        model = Sequential()
        model.add(Conv2D(30, (10, 10), input_shape=(100,100, 1), padding='same', activation= "relu"))
        model.add(MaxPooling2D((3, 3), strides=2))
        model.add(Conv2D(60, (5, 5),activation="relu", strides=1))
        model.add(MaxPooling2D((3,3)))
        model.add(Conv2D(90, (3, 3),activation="relu", strides=1 ))
        model.add(MaxPooling2D((3,3)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        self.model = model

    def fit(self, X_train, y_train, validation_data, lr, epochs, batch_size, verbose):
        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr), metrics=['accuracy'])
        self.result = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, \
                                     validation_data=validation_data, verbose=verbose)
        return self



