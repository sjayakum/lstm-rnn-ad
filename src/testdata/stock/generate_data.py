# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:15:12 2016

@author: suraj
"""


#Make a list of 6064 numbers between 4-6
import random
import numpy as np
import pickle

x_attributes = np.zeros((216,12,2))
y_attributes = np.zeros((216))


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

for i in range(216):
    temp_list = []
    place_to_insert_data = i%12
    temp_list.append(float(a_max - a_list[i])/a_range)
    temp_list.append(float(b_list[i])/b_max)
    x_attributes[i][place_to_insert_data] = temp_list
    y_attributes[i] = float(b_list[(i+1)%216]/float(b_max))
    
    
    
print x_attributes.shape
print x_attributes[0][0]
print x_attributes[0]

print x_attributes[1]

print y_attributes.shape
print y_attributes[0]
print y_attributes[1]

print y_attributes[212]
pickle.dump(x_attributes,open('x_att.p','wb'))
pickle.dump(y_attributes,open('y_att.p','wb'))

print "hi"













