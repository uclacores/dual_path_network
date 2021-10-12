import keras
import numpy as np 


import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import *

from keras import backend as K


import numpy as np
#np.random.seed(2018) # for reproducibility

from keras.utils.np_utils import to_categorical   
import os
import sys


from frm_nn_functions import *




def create_model_gru(n_classes,pkt_size=128):
    dr = 0.5 # dropout rate (%)
    in_shp = [pkt_size,2]
    sig_in = Input(in_shp)
    x = sig_in
    # x= Permute((2,1))(x)
    x = CuDNNGRU(150, input_shape=in_shp,  return_sequences=True)(x)
    # x = Dropout(0.4)(x)  # Dropout overfitting

    # model.add(GRU(layers[2],activation='tanh', return_sequences=True))
    # model.add(Dropout(0.2))  # Dropout overfitting

    x = CuDNNGRU(150, return_sequences=False)(x)
    # x = Dropout(0.4)(x)  # Dropout overfitting

    x = Dense(64)(x)
    x = Dense(n_classes)(x)
    x = Activation("softmax")(x)
    model = Model(inputs = sig_in, outputs = x)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(0.0006, clipnorm=1.),metrics=[keras.metrics.categorical_accuracy])
    model.summary()
    return model
