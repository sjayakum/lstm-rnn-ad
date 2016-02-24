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
for i in range(4032):
    attachRateList.append(random.uniform(4,6))


attachRateList = np.array(attachRateList)
#mean = np.mean(attachRateList)
#stddev = np.std(attachRateList)

encoded_attach_rate_list = attachRateList
#
#for each_element in attachRateList:
#    temp = (each_element - mean)/stddev
#    encoded_attach_rate_list.append(temp)

#encoded_attach_rate_list = np.array(encoded_attach_rate_list)


x_attributes = np.zeros((4032,288,2))
y_attributes = np.zeros((4032))

#day_mean = 3
#day_stddev = 2

#time_mean = 143.5
#time_stddev = 83.137937589686857
for i in range(4032):
    place_to_insert_data = i%288
    assign_array = []
    day_number = i%6
    time_number= i%288
    
    assign_array.append(float(day_number)/6)
    #assign_array.append(float(day_number-day_mean)/day_stddev)
    #assign_array.append(float(time_number-time_mean)/time_stddev)
    assign_array.append(float(encoded_attach_rate_list[i])/6)
    x_attributes[i][place_to_insert_data] = assign_array
    y_attributes[i] = encoded_attach_rate_list[(i+1)%4032]

print x_attributes.shape
print x_attributes[0][0]
print x_attributes[0]


print y_attributes.shape
print y_attributes[0]
pickle.dump(x_attributes,open('x_att.p','wb'))
pickle.dump(y_attributes,open('y_att.p','wb'))















