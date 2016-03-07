# -*- coding: utf-8 -*-
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
tsteps = 24
batch_size = 1
epochs = 10
attsize = 2
k = 0

inputs = pickle.load(open('X.p'))
expected_outputs = pickle.load(open('y.p'))
predicted_outputs =  np.zeros((len(inputs[720:]),8))

test_inps = inputs[720:]
test_outs = expected_outputs[720:]


model = model_from_json(open('lstm_y_8_inet.json').read())
model.load_weights('lstm_y_8_inet_weights.h5')

scores = model.evaluate(test_inps,test_outs, show_accuracy=True, verbose=1, batch_size=1)
print('RNN test score:', scores[0])

for i in range(len(inputs[720:])):
    predicted_outputs[i] = model.predict(np.array([test_inps[i]]), verbose=1, batch_size=1)
    
preds = [out[k] for out in predicted_outputs[200:250]]

errors_full = [(test_outs[i][k] - predicted_outputs[i][k]) for i in range(len(inputs[720:])) ]
errors = errors_full[200:250]
print 'avg : ',sum(errors_full)/float(len(errors_full))
print 'max error :  ', max(errors_full)
print 'min error : ',min(errors_full)

print 'avg : ',sum(preds)/float(len(preds))
print 'max pred : ', max(preds)
print 'min pred : ', min(preds)

print "*****************  Anomalies  **************************"
actual_anomalies = []
pred_anomalies = []
for i in range(len(test_inps[200:250])):
    if test_inps[i][i%12][1]*9. >= 9.:
        print 200+i, test_inps[i][i%12][1]*9., preds[i]*9.
        actual_anomalies.append(test_inps[i][i%12][1]*9.)
        pred_anomalies.append(preds[i]*9.)
    else:
        actual_anomalies.append(0)
        pred_anomalies.append(0)



print "*******************   Non-Anomalies   *********************************"


for i in range(len(test_inps[200:250])):
    if test_inps[i][i%12][1]*9. <= 9.:
        print 200+i, test_inps[i][i%12][1]*9., preds[i]*9.

test_outs_first_val = [out[k] for out in test_outs]
'''
plt.subplot(3,1,1)

plt.bar(range(len(test_outs_first_val[200:250])),test_outs_first_val[200:250],label='Expected',color='#F4561D')
plt.bar(range(len(test_outs_first_val[200:250])),preds,label='Predicted',color='#F1BD1A')

plt.legend(('Expected', 'Predicted'), loc='best')
plt.title('Expected vs Predicted')
'''


for k in range(8):
    plt.subplot(8,1,1+k)
    errors_full = [(test_outs[i][k] - predicted_outputs[i][k]) for i in range(len(inputs[720:])) ]
    plt.plot(range(len(errors)),errors_full[200:250], 'o-', label='error')
    

plt.show()


