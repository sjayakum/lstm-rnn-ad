from keras.layers.core import Dense
from keras.layers.core import TimeDistributedDense

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
from keras.layers.core import TimeDistributedMerge

# CONSTANTS
tsteps = 24
batch_size = 1
epochs = 10
attsize = 2

inputs = pickle.load(open('X.p'))
expected_outputs = pickle.load(open('y.p'))
predicted_outputs = 0

train_inps = inputs[:720]
train_outs = expected_outputs[:720]

"""
I/P               dim = 1*24*2

LSTM              dim = 1*24

TDD               dim = 1*24*8

TDM               dim = 1*8
"""

model = Sequential()
model.add(LSTM(tsteps, batch_input_shape=(batch_size,tsteps,attsize),return_sequences=True))
model.add(TimeDistributedDense(8))
model.add(TimeDistributedMerge('sum'))
model.compile(loss='mse', optimizer='rmsprop')

print "Network Built Sucessfully"


print "Training"
for j in range(epochs):
    for i in range(len(train_inps)):
        model.fit(np.array([train_inps[i]]), np.array([train_outs[i]]),verbose=1,nb_epoch=j)

print "Finished Training"


open('lstm_y_8_inet.json','w').write(model.to_json())
model.save_weights('lstm_y_8_inet_weights.h5')