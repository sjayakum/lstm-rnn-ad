import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.models import Graph
from keras.models import model_from_json
import pickle


#CONSTANTS
tsteps = 24
batch_size = 1
epochs = 20
attsize = 2



inputs = pickle.load(open('xAttn.p'))
expected_outputs = pickle.load(open('yAttn.p'))
predicted_outputs = 0

train_inps = inputs[:1200]
train_outs = expected_outputs[:1200]

test_inps = inputs[1200:]
test_outs = expected_outputs[1200:]



model = Sequential()
model.add(LSTM(tsteps, batch_input_shape=(batch_size,tsteps,attsize),return_sequences=True))
model.add(TimeDistributedDense(1))
model.add(LSTM(tsteps, batch_input_shape=(batch_size,tsteps,attsize)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='rmsprop')


print "Training"

for j in range(epochs):
    for i in range(len(train_inps)):
        model.fit(train_inps[i], np.array([train_outs[i]]),verbose=1,nb_epoch=1)


print "Finished Training"

#
# open('lstm_inet_tdd_n_20.json','w').write(model.to_json())
# model.save_weights('lstm_inet_tdd_weights_n_20.h5')
#
# #
# #



#
# model = model_from_json(open('lstm_inet_tdd_n_20.json').read())
# model.load_weights('lstm_inet_tdd_weights_n_20.h5')
#

def ohe_predicted_value(previous_input, value, ):
    """
    :param previous_input: get previous input to know the index
    :param value: predicted value from the model
    :return: numpy array of dimension 1*24*2


    """
    dim23_array = np.zeros((24, 2))
    previous_input = previous_input[0]
    i = 0
    while (previous_input[i][1] == 0):
        i = i + 1

    # now i will tell which position from 0-23 i was in previous input



    # special case
    if (i == 23):
        dim23_array[0][0] = (previous_input[i][0] + 1) % 7
        dim23_array[0][1] = value
        pass
    else:
        # now i is the next time-step of previous input [or current time-step]
        dim23_array[i][0] = previous_input[i][0]
        dim23_array[i][1] = value

    # change dim from 24*2 to 1*24*2 and return
    return np.array([dim23_array])


preds1 = []
preds1.append(0)

preds2 = []
preds2.append(0)
preds2.append(0)



corrected_test_outs = []
corrected_test_outs.append(0)

corrected_test_outs.extend(test_outs)

for i in range(len(test_inps)):
    # print "****************************"
    # print "****First Input to Model****"
    # print test_inps[i]
    pred = model.predict(test_inps[i], batch_size=1)
    # print "****First Predicted Value*****"
    # print pred[0][0]
    # print "****First Actual Value******"
    # print test_outs[i]
    preds1.append(pred[0][0])
    # preds2.append(pred[0][0])
    input2_model = ohe_predicted_value(test_inps[i], pred[0][0])
    # print "****Second Input to Model****"
    # print input2_model
    pred2 = model.predict(input2_model, batch_size=1)
    # print "****Second Predicted Value****"
    # print pred2[0][0]
    # print "****Second Actual Value****"
    # print test_outs[i+1]
    preds2.append(pred2[0][0])



plt.subplot(3, 1, 1)
plt.plot(corrected_test_outs)
plt.title('Expected')
plt.subplot(3, 1, 2)
plt.plot(np.array(preds1))
plt.title('Predicted')
plt.subplot(3, 1, 3)
plt.plot(np.array(preds2))
plt.title('Predicted2')

plt.show()
