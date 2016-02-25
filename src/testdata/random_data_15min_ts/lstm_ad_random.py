# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Graph
from keras.models import model_from_json
from pandas import DataFrame
import pandas as pd
import pickle

# CONSTANTS
tsteps = 12
batch_size = 1
epochs = 20
attsize = 3

inputs = pickle.load(open('x_att.p'))
expected_outputs = pickle.load(open('y_att.p'))
predicted_outputs = 0

train_inps = inputs[:2688]
train_outs = expected_outputs[:2688]


model = Sequential()
model.add(LSTM(tsteps, batch_input_shape=(batch_size,tsteps,attsize)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

print "Network Built Sucessfully"


print "Training"
for j in range(epochs):
    for i in range(len(train_inps)):
        model.fit(np.array([train_inps[i]]), np.array([train_outs[i]]),verbose=1,nb_epoch=j)


print "Finished Training"


open('lstm_inet_dense_ad_random_15min_20_epch.json','w').write(model.to_json())
model.save_weights('lstm_inet_weights_dense_ad_random_15min_20_epch.h5')

