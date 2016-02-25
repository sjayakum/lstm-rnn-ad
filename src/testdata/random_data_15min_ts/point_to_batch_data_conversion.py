# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:21:20 2016

@author: suraj
"""



import pickle
import numpy as np
X = pickle.load(open('x_att.p'))
y = pickle.load(open('y_att.p'))


batchX = []
batchy = []


def convertPointsToBatch(day_of_week,data1,data2):
    
    for i in range(5):
        batchX.extend(data1[((i*672)+((day_of_week)*96)):((i*672)+((day_of_week)*96))+96])
        batchy.extend(data2[((i*672)+((day_of_week)*96)):((i*672)+((day_of_week)*96))+96])
        
    pass


for i in range(7):
    convertPointsToBatch(i,X,y)

batchX = np.array(batchX)
batchy = np.array(batchy)
print batchX.shape
print batchy.shape
print batchX[0]
print batchy[0]
pickle.dump(batchX,open('batch_x_att.p','wb'))
pickle.dump(batchy,open('batch_y_att.p','wb'))