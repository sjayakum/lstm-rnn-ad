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

# CONSTANTS
tsteps = 12
batch_size = 1
epochs = 10
attsize = 2


inputs = pickle.load(open('test_x_natt.p'))
expected_outputs = pickle.load(open('test_y_natt.p'))
predicted_outputs = 0


test_inps = inputs
test_outs = expected_outputs


model = model_from_json(open('lstm_inet_dense_ad_stock.json').read())
model.load_weights('lstm_inet_weights_dense_ad_stock.h5')



def ohe_predicted_value(previous_input, value, ):
    """
    :param previous_input: get previous input to know the index
    :param value: predicted value from the model
    :return: numpy array of dimension 1*288*2

    """
    dim23_array = np.zeros((12, 2))
    previous_input = previous_input[0]
    i = 0
    while (previous_input[i][1] == 0):
        i = i + 1
        
 
    # now i will tell which position from 0-11 i was in previous input


    # special case
    if (i == 11):
        dim23_array[0][0] = (1995. - (1995. - (previous_input[i][0]*24.)))/24.
        #dim23_array[0][1] = (previous_input[i][1] + 1) % 288
        dim23_array[0][1] = value
        pass
    else:
        # now i is the next time-step of previous input [or current time-step]
        dim23_array[i+1][0] = (previous_input[i][0])
        #dim23_array[i][1] = (previous_input[i][1] + 1)
        dim23_array[i+1][1] = value

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
d2= {k+1:[0]*(k+1) for k in range(n)}

corrected_test_outs = []
corrected_test_outs.append(0)

corrected_test_outs.extend(test_outs)

preds1 = []
preds2 = []


for i in range(len(test_inps)):
   
    predict_n_further(i, n,np.array([test_inps[i]]), model, batch_size=1)
  
df = DataFrame(d1)
print df


plt.subplot(9, 1, 1)
plt.plot(corrected_test_outs)
plt.title('Expected')
for i in range(n):
    plt.subplot(9, 1, i+2)
    plt.plot(np.array(d2[i+1]))
    plt.title('Predicted ' + str(i+1))

plt.show()





