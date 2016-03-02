# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:43:33 2016

@author: suraj
"""

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Graph
from keras.models import model_from_json
from pandas import DataFrame
import pandas as pd
import pickle



inputs = pickle.load(open('x_att.p'))
expected_outputs = pickle.load(open('y_att.p'))
predicted_outputs = 0

train_inps = inputs[:2688]
train_outs = expected_outputs[:2688]


model = Sequential()
#model.add(LSTM(tsteps, batch_input_shape=(batch_size,tsteps,attsize)))
model.add(Dense(2,input_shape=(6,)))

model.add(Activation('tanh'))
model.compile(loss='mse', optimizer='rmsprop')

print "Network Built Sucessfully"


print "Training"
for j in range(10):
    for i in range(len(train_inps)):
        model.fit(np.array([train_inps[i]])  , np.array([train_outs[i]]), verbose=1,nb_epoch=j)


print "Finished Training"


open('dense_fourier.json','w').write(model.to_json())
model.save_weights('weights_dense_fourier.h5')

