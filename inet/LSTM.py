import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.core import Dropout
from keras.models import  Graph
from keras.models import model_from_json
import pickle




#CONSTANTS


tsteps = 24
batch_size = 1
epochs = 10
attsize = 2


inputs = pickle.load(open('xAttn.p'))
expected_outputs = pickle.load(open('yAttn.p'))
predicted_outputs = 0

train_inps = inputs[:1200]
train_outs = expected_outputs[:1200]

test_inps = inputs[1200:]
test_outs = expected_outputs[1200:]

model = Sequential()
# model.add(LSTM(tsteps, batch_input_shape=(batch_size,tsteps,attsize)))
model.add(Dense(tsteps))
# model.add(Activation('tanh'))
# model.add(Dropout(0.2))
model.compile(loss='mse', optimizer='sgd')

print "Network Built Sucessfully"


print "Training"

for j in range(epochs):
    for i in range(len(train_inps)):
        model.fit(train_inps[i], np.array([train_outs[i]]),verbose=1,nb_epoch=1)


print "Finished Training"


open('lstm_inet_dense.json','w').write(model.to_json())
model.save_weights('lstm_inet_weights_dense.h5')

'''
model = model_from_json(open('lstm_inet_dense.json').read())
model.load_weights('lstm_inet_weights_dense.h5')
'''

#
# sample_outs = []
#
# act_count = 0
# pred_count = 0
#
#     # preds added to chain outputs to inputs, along with the if else
# preds = []
#
# tp = 0
#
# for i in range(len(test_outs)):
#     pred = model.predict(test_inps[i],batch_size=1)
#     # temp.append(pred[0][0])
#     preds.append(pred[0][0])
#     # import pdb;pdb.set_trace()
#     # print test_expected_outputs[i][j]-pred[0][0]
#     diff = abs(test_outs[i]-pred[0][0])
#     deviation = float(diff)/len(test_outs) * 100
#     if deviation < 5.:
#         pred_count += 1
#     print '***********************************'
#     print '\t\titer ' + str(i)
#     print 'Actual : ',test_outs[i]
#     print 'Predicted : ', pred[0][0]
#     print '***********************************'
#
# # import pdb;pdb.set_trace()
# print 'Correct predictions : ',pred_count
# print 'Testing accuracy : ', float(pred_count)/len(test_outs) * 100
#
# plt.subplot(2, 1, 1)
# plt.plot(test_outs)
# plt.title('Expected')
# plt.subplot(2, 1, 2)
# plt.plot(np.array(preds))
# plt.title('Predicted')
# plt.show()
