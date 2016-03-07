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
import os

#os.system('python lstm_ad_random.py')
# CONSTANTS
tsteps = 12
batch_size = 1
epochs = 10
attsize = 3
k = 0

online_inps = pickle.load(open('online_x_att.p'))
online_outs = pickle.load(open('online_y_att.p'))

model = model_from_json(open('lstm_y_8.json').read())
model.load_weights('lstm_y_8_weights.h5')

print "Training"
for j in range(epochs):
    for i in range(len(online_inps)):
        model.fit(np.array([online_inps[i]]), np.array([online_outs[i]]),verbose=1,nb_epoch=j)

print "Finished Training"


open('lstm_y_8_online.json','w').write(model.to_json())
model.save_weights('lstm_y_8_online_weights.h5')
