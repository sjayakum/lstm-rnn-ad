# -*- coding: utf-8 -*-
import pickle
import numpy as np
examples = open('inet.csv').read().strip().split('\n')
daycounter = 0
attList = []
actualDofMonth = 0
for ex in examples:
    tempList = []
    y = ex.split(',')[1]
    actualDofMonth = ((daycounter) /24 %7)
    tempList.append(actualDofMonth)
    tempList.append(float(y)/150000)
    attList.append(tempList)
    daycounter = daycounter + 1

print len(attList)

attList = np.array(attList)

X = np.zeros((1512,24,2))
y = np.zeros((1512,8))

for i in range(len(attList)):
    X[i][i%24][0] = attList[i][0]
    X[i][i%24][1] = attList[i][1]
#print X, X.shape, X[0], X[1]

for i in range(len(X)):
    for j in range(8):
        y[i][j] = X[(i+j+1)%len(X)][(i+j+1)%24][1]

pickle.dump(X,open('X.p','wb'))
pickle.dump(y,open('y.p','wb'))