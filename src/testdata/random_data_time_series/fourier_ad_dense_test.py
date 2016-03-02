# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:28:22 2016

@author: suraj
"""
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
#import os


#os.system('python lstm_ad_random.py')
# CONSTANTS
tsteps = 12
batch_size = 1
epochs = 10
attsize = 3

inputs = pickle.load(open('x_att.p'))
expected_outputs = pickle.load(open('y_att.p'))
predicted_outputs = []

test_inps = inputs[2688:2688+97]
test_outs = expected_outputs[2688:2688+97]


model = model_from_json(open('dense_fourier.json').read())
model.load_weights('weights_dense_fourier.h5')


freq_x_axis = np.fft.fftfreq(len(test_inps))

for i in range(len(test_inps)):
    predicted_outputs.append(model.predict(np.array([test_inps[i]]))[0])


converted_expected = []
converted_predicted = []



a = [np.complex(test_outs[i][0], test_outs[i][1]) for i in range(len(test_outs))]
b = np.fft.ifft(a)

temp_complex = []

for i in range(len(test_inps)):
    temp_complex.append(np.array([np.complex(predicted_outputs[i][0],predicted_outputs[i][1])]))




temp_complex = []

for i in range(len(test_inps)):
    temp_complex.append(np.array([np.complex(predicted_outputs[i][0],predicted_outputs[i][1])]))

converted_predicted.append(np.array([np.complex(0,0)]))
converted_predicted.extend(np.fft.ifft(np.array(temp_complex)))
converted_predicted = np.array(converted_predicted)

print "hi"
plt.plot(b.real,label='Expected')
plt.plot(converted_predicted.real,label='Predicted')
plt.legend(loc='best')
plt.title('Expected vs Predicted Attach Rates for Test Week (Batch)')
plt.xlabel('Frequency')
plt.ylabel('Attach Rate')
plt.show()

#plt.savefig('LSTM_12_ts_10_epch_batch_mon.png')



