# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:15:12 2016

@author: suraj
"""


#Make a list of 6064 numbers between 4-6
import random
import numpy as np
import pickle

attachRateList = []
for i in range(3360):
    attachRateList.append(random.uniform(4,6))


attachRateList = np.array(attachRateList)
encoded_attach_rate_list = attachRateList

x_attributes = np.zeros((3360,12,3))
y_attributes = np.zeros((3360))


for i in range(3360):
    place_to_insert_data = i%12
    
    assign_array = []
    day_number = i%7
    time_number= i%96
    assign_array.append(float(day_number)/6)
    assign_array.append(float(time_number)/96)
    assign_array.append(float(encoded_attach_rate_list[i])/6)
    x_attributes[i][place_to_insert_data] = assign_array
    y_attributes[i] = float(encoded_attach_rate_list[(i+1)%3360])/6.

print x_attributes.shape
print x_attributes[0][0]
print x_attributes[0]


print y_attributes.shape
print y_attributes[0]
pickle.dump(x_attributes,open('x_att.p','wb'))
pickle.dump(y_attributes,open('y_att.p','wb'))















