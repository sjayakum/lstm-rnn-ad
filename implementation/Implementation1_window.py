import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import  Graph
from keras.models import model_from_json
import pickle




#5 min interval
tsteps = 36
batch_size = 1
epochs = 4
attsize = 2


def generate_data():
    x = pickle.load(open('mondayXnew.p'))
    return x

def generate_actual():
    y = pickle.load(open('mondayYnew.p'))
    return y

#GET X,y and y'

input_data = generate_data()
expected_outputs = generate_actual()
predicted_outputs = 0

test_data = pickle.load(open('tesmondayXnew.p'))
test_expected_outputs  = pickle.load(open('Data/tesmondayYnew.p'))

print "All Data Generated"

model = Sequential()
#CHANGE made: (ash) -> LSTM(batch_size*tsteps*attsize*2)
# model.add(LSTM(batch_size*tsteps, batch_input_shape=(batch_size, tsteps, attsize)))
model.add(LSTM(36, batch_input_shape=(1,36,2)))
#CHANGE made: (ash) -> Dense(1)
model.add(Dense(36))
model.compile(loss='mse', optimizer='rmsprop')

print "Network Built"
'''
x = np.array([[[0.,4.5]]*36])
x.shape = (1,36,2)
y = np.array([[4.5]*36])
y.shape = (1,36)
'''
print "Training"
# import pdb;pdb.set_trace()
wind_count = 0
for i in range(4):
    for j in range(288):
        x = np.array([[[0.,0.]]*36])
        x[0][wind_count] = np.array(input_data[i][j])
        y = np.array([[0.]*36])
        y[0][wind_count] = np.array(expected_outputs[i][j])
        wind_count += 1
        if wind_count == 36:
            wind_count = 0
        model.fit(x,y,verbose=1)

# pickle.dump(model,open('model_1_lstm.p','wb'))
# model = pickle.load(open('model_1_lstm.p'))
open('lstm_2_wind.json','w').write(model.to_json())
model.save_weights('lstm_2_wind_weights.h5')

model = model_from_json(open('lstm_2_wind.json').read())
model.load_weights('lstm_2_wind_weights.h5')
test_predicted_outputs = model.predict(np.array([[test_data[0][0]]]),batch_size=1)
test_outs = []

act_count = 0
pred_count = 0
tp = 0
'''
for i in range(len(test_expected_outputs)):
    temp = []
    for j in range(len(test_expected_outputs[i])):
        pred = model.predict(np.array([[test_data[i][j]]]),batch_size=1)
        temp.append(pred[0][0])
        # import pdb;pdb.set_trace()
        # print test_expected_outputs[i][j]-pred[0][0]
        diff = abs(test_expected_outputs[i][j]-pred[0][0])
        deviation = float(diff)/test_expected_outputs[i][j] * 100
        if deviation >= 20.:
            pred_count += 1
            if test_data[i][j][1] > 6.:
                tp += 1
    test_outs.append(temp)

for i in range(len(test_data)):
    for j in range(len(test_data[i])):
        if test_data[i][j][1] > 6.:
            act_count += 1

print 'Anomalies predicted : ',pred_count
print 'Anomalies expected : ',act_count
print 'Testing accuracy : ', pred_count/float(act_count) * 100
print 'Anomalies correctly detected : ', tp
print 'Precision : ', tp/float(pred_count)
# print count/float((len(test_expected_outputs)*len(test_expected_outputs[0]))) * 100

# import pdb;pdb.set_trace()

plt.subplot(2, 1, 1)
plt.plot(test_expected_outputs)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(np.array(test_outs))
plt.title('Predicted')
plt.show()
'''

