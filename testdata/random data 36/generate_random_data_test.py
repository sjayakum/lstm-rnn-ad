# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:12:08 2016

@author: suraj
"""

import random
import numpy as np
import pickle

attachRateList = []
for i in range(2016):
    attachRateList.append(random.uniform(4,6))


attachRateList = np.array(attachRateList)
mean = np.mean(attachRateList)
stddev = np.std(attachRateList)

encoded_attach_rate_list = []

for each_element in attachRateList:
    temp = (each_element - mean)/stddev
    encoded_attach_rate_list.append(temp)

encoded_attach_rate_list = np.array(encoded_attach_rate_list)



x_attributes = np.zeros((2016,36,3))
y_attributes = np.zeros((2016))

day_mean = 3
day_stddev = 2

time_mean = 143.5
time_stddev = 83.137937589686857

for i in range(2016):
    probability_variable = random.random()
    place_to_insert_data = i%32
    assign_array = []
    day_number = i%6
    time_number= i%288
    assign_array.append(float(day_number-day_mean)/day_stddev)
    assign_array.append(float(time_number-time_mean)/time_stddev)
    
#    if(probability_variable > 0.75):
#        assign_array.append((random.uniform(6,10)-mean)/stddev)
#    else:
    assign_array.append(encoded_attach_rate_list[i])
        
        
    x_attributes[i][place_to_insert_data] = assign_array
    y_attributes[i] = encoded_attach_rate_list[(i+1)%2016]
    
    
print x_attributes.shape
print x_attributes[0][0]
print x_attributes[0]


print y_attributes.shape
print y_attributes[0]
pickle.dump(x_attributes,open('test_x_natt.p','wb'))
pickle.dump(y_attributes,open('test_y_natt.p','wb'))


