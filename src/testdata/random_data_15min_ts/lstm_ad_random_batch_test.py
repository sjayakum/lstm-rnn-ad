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

inputs = pickle.load(open('batch_x_att.p'))
expected_outputs = pickle.load(open('batch_y_att.p'))
predicted_outputs = 0

test_inps = inputs[2688:]
test_outs = expected_outputs[2688:]


model = model_from_json(open('lstm_inet_dense_ad_random_15min_batch.json').read())
model.load_weights('lstm_inet_weights_dense_ad_random_15min_batch.h5')




def ohe_predicted_value(previous_input, value, ):
    """
    :param previous_input: get previous input to know the index
    :param value: predicted value from the model
    :return: numpy array of dimension 1*12*3

    """
    dim23_array = np.zeros((12, 3))
    previous_input = previous_input[0]
    i = 0
    while (previous_input[i][2] == 0):
        i = i + 1
        
 
    # now i will tell which position from 0-11 i was in previous input
    # special case
    if (i == 11):
        if previous_input[i][1] == 1:
                dim23_array[0][0] = (previous_input[i][0] + 1)%7
        else:
            dim23_array[0][0] = (previous_input[i][0])
            
        dim23_array[0][1] = ((previous_input[i][1]*96 + 1) % 96)/96.
        dim23_array[0][2] = value
    else:
        # now i is the next time-step of previous input [or current time-step]
        dim23_array[i+1][0] = (previous_input[i][0])
        dim23_array[i+1][1] = previous_input[i][1]        
        dim23_array[i+1][2] = value

    # change dim from 288*2 to 1*288*2 and return
    return np.array([dim23_array])


def predict_n_further(index, n, inp, model, batch_size=1):
    '''
    :param index: test example index
    :param n: number of timesteps into the future to predict
    :param inp: inp value
    :param model: network model
    :param batch_size: 1
    '''
    for i in range(n):
        if i == 0:
            pred = model.predict(inp, batch_size)
	    d2[i+1].append(pred[0][0])
            next_inp = ohe_predicted_value(inp,pred[0][0])
        else:
            # prev_inp = next_inp
            pred = model.predict(next_inp,batch_size)
	    d2[i+1].append(pred[0][0])
            next_inp = ohe_predicted_value(next_inp,pred[0][0])
        d1[index].append(pred[0][0])

n = 8

d1= {k:[] for k in range(len(test_inps))}
d2= {k+1:[] for k in range(n)}

corrected_test_outs = []

corrected_test_outs.extend(test_outs)

preds1 = []
preds2 = []


for i in range(len(test_inps)):
   
    predict_n_further(i, n,np.array([test_inps[i]]), model, batch_size=1)
  
df = DataFrame(d1)
print df

'''
plt.subplot(2, 1, 1)
plt.plot(corrected_test_outs)
plt.title('Expected')
for i in range(1):
    plt.subplot(2, 1, i+2)
    plt.plot(np.array(d2[i+1]))
    plt.title('Predicted ' + str(i+1))
'''
plt.plot(range(len(corrected_test_outs)),corrected_test_outs,label='Expected')
plt.plot(range(len(d2[2])),d2[2],label='Predicted')
plt.legend(loc='best')
plt.title('Expected vs Predicted Attach Rates for Test Week (Batch)')
plt.xlabel('Time Step')
plt.ylabel('Attach Rate')
plt.show()

#plt.savefig('LSTM_12_ts_10_epch_batch_mon.png')



