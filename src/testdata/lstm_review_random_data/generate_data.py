# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:55:52 2016

@author: suraj
"""

import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

attachRateList = []


for i in range(672):
    attachRateList.append(random.uniform(4.1,5.9))
    

attachRateList = np.array(attachRateList)
encoded_attach_rate_list = attachRateList

x_attributes = np.zeros((2688,12,3))
y_attributes = np.zeros((2688))
test_x_attributes = np.zeros((672,12,3))
test_y_attributes = np.zeros((672))

plotting_list1 = []
plotting_list2 = []
plotting_list3 = []
plotting_list4 = []
plotting_list5 = []

for i in range(672):
    place_to_insert_data = i%12
    #Create temp array
    assign_array = []
    day_number = i%7
    time_number= i%96
    assign_array.append(float(day_number)/6)
    assign_array.append(float(time_number)/96)
    assign_array.append(float(encoded_attach_rate_list[i])/6)
    #assign it  to x-attributes & y-attrivutes
    x_attributes[i][place_to_insert_data] = assign_array
    y_attributes[i] = float(encoded_attach_rate_list[(i+1)%672])/6.
    #just for plotting
    plotting_list1.append(float(encoded_attach_rate_list[i])/6)


#
# for i in range(2016):
#    place_to_insert_data = (i+672)%12
#    assign_array = []
#    day_number = (i+672)%7
#    time_number= (i+672)%96
#    assign_array.append(float(day_number)/6)
#    assign_array.append(float(time_number)/96)
#
#
#    plus_minus_prob = random.random()
#
#
#    #65% probability of Minus & 35% probability of Plus
#    if(plus_minus_prob<0.65):
#        #minus
#        assign_array.append((float(encoded_attach_rate_list[(i%672)])/6)+(-1)*random.uniform(0.05,0.09))
#
#        #just used for plotting
#        if(i<672):
#            plotting_list2.append((float(encoded_attach_rate_list[(i%672)])/6)+(-1)*random.uniform(0.05,0.09))
#        elif(672<i<1344):
#            plotting_list3.append((float(encoded_attach_rate_list[(i%672)])/6)+(-1)*random.uniform(0.05,0.09))
#        elif(1344<i<(1344+672)):
#            plotting_list4.append((float(encoded_attach_rate_list[(i%672)])/6)+(-1)*random.uniform(0.05,0.09))
#
#
#        pass
#    else:
#        #plus
#        assign_array.append((float(encoded_attach_rate_list[(i%672)])/6)+random.uniform(0.05,0.09))
#
#         #just used for plotting
#        if(i<672):
#            plotting_list2.append((float(encoded_attach_rate_list[(i%672)])/6)+random.uniform(0.05,0.09))
#        elif(672<i<1344):
#            plotting_list3.append((float(encoded_attach_rate_list[(i%672)])/6)+random.uniform(0.05,0.09))
#        elif(1344<i<(1344+672)):
#            plotting_list4.append((float(encoded_attach_rate_list[(i%672)])/6)+random.uniform(0.05,0.09))
#
#        pass
#
#
#
#    x_attributes[(i+672)][place_to_insert_data] = assign_array
#    y_attributes[(i+672)] =float(encoded_attach_rate_list[(i+1)%672])/6.
#
#
#
#GENEREATE TEST DATA
    
for i in range(672):
    place_to_insert_data = i%12
    #Create temp array
    assign_array = []
    day_number = i%7
    time_number= i%96
    assign_array.append(float(day_number)/6)
    assign_array.append(float(time_number)/96)
    
    plus_minus_prob = random.random()
    anomaly_prob = random.random()    
    if(plus_minus_prob<0.65):
        assign_array.append((float(encoded_attach_rate_list[(i%672)])/6)+(-1)*random.uniform(0.05,0.09))
        plotting_list5.append((float(encoded_attach_rate_list[(i%672)])/6)+(-1)*random.uniform(0.05,0.09))
    else:
        assign_array.append((float(encoded_attach_rate_list[(i%672)])/6)+random.uniform(0.05,0.09))
        plotting_list5.append((float(encoded_attach_rate_list[(i%672)])/6)+random.uniform(0.05,0.09))
        
            
    #assign it  to x-attributes & y-attrivutes
    test_x_attributes[i][place_to_insert_data] = assign_array
    test_y_attributes[i] = float(encoded_attach_rate_list[(i+1)%672])/6.
    #just for plotting

for i in range(672):
    place_to_insert_data = i%12
    prob_var = random.random()
    if(prob_var>0.95):
        test_x_attributes[i][place_to_insert_data][2] = random.uniform(7,10)/6.
        pass
    
    

    
plt.subplot(5, 1, 1)
plt.plot(np.array(plotting_list1))
plt.subplot(5, 1, 2)
plt.plot(np.array(plotting_list2))
plt.subplot(5, 1, 3)
plt.plot(np.array(plotting_list3))
plt.subplot(5, 1, 4)
plt.plot(np.array(plotting_list4))
plt.subplot(5, 1, 5)
plt.plot(np.array(plotting_list5))
plt.show()

#
#pickle.dump(x_attributes,open('x_att.p','wb'))
#pickle.dump(y_attributes,open('y_att.p','wb'))
#
pickle.dump(test_x_attributes,open('test_x_att.p','wb'))
pickle.dump(test_y_attributes,open('text_y_att.p','wb'))














