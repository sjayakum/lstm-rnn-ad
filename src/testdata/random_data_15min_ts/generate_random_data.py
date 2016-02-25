# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:15:12 2016

@author: suraj
"""


#Make a list of 6064 numbers between 4-6
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

attachRateList = []
for i in range(3360):
    attachRateList.append(random.uniform(4,6))


attachRateList = np.array(attachRateList)
encoded_attach_rate_list = attachRateList

x_attributes = np.zeros((3360,12,3))
y_attributes = np.zeros((3360))

plotting_list1 = []
plotting_list2 = []
plotting_list3 = []
plotting_list4 = []
plotting_list5 = []

for i in range(672):
    place_to_insert_data = i%12
    assign_array = []
    day_number = i%7
    time_number= i%96
    assign_array.append(float(day_number)/6)
    assign_array.append(float(time_number)/96)
    plotting_list1.append(float(encoded_attach_rate_list[i])/6)
    assign_array.append(float(encoded_attach_rate_list[i])/6)
    x_attributes[i][place_to_insert_data] = assign_array
    y_attributes[i] = float(encoded_attach_rate_list[(i+1)%3360])/6.

print x_attributes.shape
print x_attributes[0][0]
print x_attributes[0]


print y_attributes.shape
print y_attributes[0]

for i in range(2688):
    place_to_insert_data = (i+672)%12
    assign_array = []
    day_number = (i+672)%7
    time_number= (i+672)%96
    assign_array.append(float(day_number)/6)
    assign_array.append(float(time_number)/96)
    
    if(i<672):
        plotting_list2.append((float(encoded_attach_rate_list[(i%672)])/6)+(-1**i)*random.uniform(0.05,0.09))    
    elif(672<i<1344):
        plotting_list3.append((float(encoded_attach_rate_list[(i%672)])/6)+(-1**i)*random.uniform(0.05,0.09))   
    elif(1344<i<(1344+672)):
        plotting_list4.append((float(encoded_attach_rate_list[(i%672)])/6)+(-1**i)*random.uniform(0.05,0.09))   
    else:
        plotting_list5.append((float(encoded_attach_rate_list[(i%672)])/6)+(-1**i)*random.uniform(0.01,0.06))      
    assign_array.append((float(encoded_attach_rate_list[(i%672)])/6)+(-1**i)*random.uniform(0.05,0.09))
    x_attributes[(i+672)][place_to_insert_data] = assign_array
    
    y_attributes[(i+672)] =float(encoded_attach_rate_list[(i%672)+1])/6.
    
plt.subplot(3, 1, 1)
plt.plot(np.array(plotting_list1[96:200]))
plt.subplot(3, 1, 2)
plt.plot(np.array(plotting_list2[96:200]))
plt.subplot(3, 1, 3)
plt.plot(np.array(plotting_list3[96:200]))
plt.show()


pickle.dump(x_attributes,open('x_att.p','wb'))
pickle.dump(y_attributes,open('y_att.p','wb'))















