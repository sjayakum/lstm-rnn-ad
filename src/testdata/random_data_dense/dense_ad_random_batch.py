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

# CONSTANTS
tsteps = 12
batch_size = 1
epochs = 10
attsize = 3

inputs = pickle.load(open('batch_x_att.p'))
expected_outputs = pickle.load(open('batch_y_att.p'))
predicted_outputs = 0

train_inps = inputs[:2688]
train_outs = expected_outputs[:2688]


model = Sequential()
#model.add(LSTM(tsteps, batch_input_shape=(batch_size,tsteps,attsize)))
model.add(Dense(1,input_shape=(1,)))

model.add(Activation('relu'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print "Network Built Sucessfully"


print "Training"
for j in range(epochs):
    for i in range(len(train_inps)):
        model.fit(np.array([[train_inps[i][i%12][2]]])  , np.array([[train_outs[i]]]), batch_size=batch_size,verbose=1,nb_epoch=j)


print "Finished Training"


open('dense_ad_random_15min_batch.json','w').write(model.to_json())
model.save_weights('weights_dense_ad_random_15min_batch.h5')

