# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:12:08 2016

@author: suraj
"""
import random
import numpy as np
import pickle

x_attributes = np.zeros((80,12,2))
y_attributes = np.zeros((80))


fobj = open('stock.csv')

a_list = []
b_list = []
for eachLine in fobj:
    a,b = eachLine.split(',')
    a_list.append(float(a.split('-')[0]))
    b_list.append(float(b))



b_max = max(b_list)
a_range = max(a_list) - min(a_list)
a_max = max(a_list)

print a_max
print a_range

for i in range(80):
    temp_list = []
    place_to_insert_data = i%12
    temp_list.append(float(a_max - a_list[216 + i])/a_range)
    temp_list.append((b_list[216+i])/b_max)
    x_attributes[i][place_to_insert_data] = temp_list
    y_attributes[i] = (b_list[(216 + i+1)%296]/float(b_max))
    
    
    
print x_attributes.shape
print x_attributes[0][0]
print x_attributes[0]

print x_attributes[1]

print y_attributes.shape
print y_attributes[0]
print y_attributes[1]


pickle.dump(x_attributes,open('test_x_natt.p','wb'))
pickle.dump(y_attributes,open('test_y_natt.p','wb'))


