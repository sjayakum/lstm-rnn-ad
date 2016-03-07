# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:11:08 2016

@author: suraj
"""


import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

attachRateList = []
online_attach_rate_list = []

for i in range(672):
    attachRateList.append(random.uniform(4.1,5.9))
    
for i in range(672):
   online_attach_rate_list.append(random.uniform(7.1,8.9))
    
attachRateList = np.array(attachRateList)
encoded_attach_rate_list = attachRateList
online_attach_rate_list = np.array(online_attach_rate_list)

x_attributes = np.zeros((2688,12,3))
y_attributes = np.zeros((2688,8))
test_x_attributes = np.zeros((672,12,3))
test_y_attributes = np.zeros((672,8))
online_x_attributes = np.zeros((672,12,3))
online_y_attributes = np.zeros((672,8))
test_online_x_attributes = np.zeros((672,12,3))
test_online_y_attributes = np.zeros((672,8))

plotting_list1 = []
plotting_list2 = []
plotting_list3 = []
plotting_list4 = []
plotting_list5 = []
plotting_list6 = []
plotting_list7 = []
'''
for i in range(672):
    place_to_insert_data = i%12
    #Create temp array
    assign_array = []
    day_number = i%7
    time_number= i%96
    assign_array.append(float(day_number)/6.)
    assign_array.append(float(time_number)/96.)
    assign_array.append(float(encoded_attach_rate_list[i])/6.)
    #assign it  to x-attributes & y-attrivutes
    x_attributes[i][place_to_insert_data] = assign_array
    y_attributes[i] = float(encoded_attach_rate_list[(i+1)%672])/6.
    #just for plotting
    plotting_list1.append(float(encoded_attach_rate_list[i])/6.)



for i in range(2016):
    place_to_insert_data = (i+672)%12
    assign_array = []
    day_number = (i+672)%7
    time_number= (i+672)%96
    assign_array.append(float(day_number)/6.)
    assign_array.append(float(time_number)/96.)


    plus_minus_prob = random.random()


    #65% probability of Minus & 35% probability of Plus
    if(plus_minus_prob<0.65):
        #minus
        assign_array.append((float(encoded_attach_rate_list[(i%672)])/6.)+(-1)*random.uniform(0.05,0.09))

        #just used for plotting
        if(i<672):
            plotting_list2.append((float(encoded_attach_rate_list[(i%672)])/6.)+(-1)*random.uniform(0.05,0.09))
        elif(672<i<1344):
            plotting_list3.append((float(encoded_attach_rate_list[(i%672)])/6.)+(-1)*random.uniform(0.05,0.09))
        elif(1344<i<(1344+672)):
            plotting_list4.append((float(encoded_attach_rate_list[(i%672)])/6.)+(-1)*random.uniform(0.05,0.09))


        pass
    else:
        #plus
        assign_array.append((float(encoded_attach_rate_list[(i%672)])/6.)+random.uniform(0.05,0.09))

         #just used for plotting
        if(i<672):
            plotting_list2.append((float(encoded_attach_rate_list[(i%672)])/6.)+random.uniform(0.05,0.09))
        elif(672<i<1344):
            plotting_list3.append((float(encoded_attach_rate_list[(i%672)])/6.)+random.uniform(0.05,0.09))
        elif(1344<i<(1344+672)):
            plotting_list4.append((float(encoded_attach_rate_list[(i%672)])/6.)+random.uniform(0.05,0.09))

        pass



    x_attributes[(i+672)][place_to_insert_data] = assign_array



for i in range(2016):
    for k in range(8):
        y_attributes[(i+672)][k] = float(x_attributes[ (i+1+k)   %2016 ][(i+1+k)%12][2]   )
        
        

#GENERATE TEST DATA
    
for i in range(672):
    place_to_insert_data = i%12
    #Create temp array
    assign_array = []
    day_number = i%7
    time_number= i%96
    assign_array.append(float(day_number)/6.)
    assign_array.append(float(time_number)/96.)
    
    plus_minus_prob = random.random()
    anomaly_prob = random.random()    
    if(plus_minus_prob<0.65):
        assign_array.append((float(encoded_attach_rate_list[(i%672)])/6.)+(-1)*random.uniform(0.05,0.09))
        plotting_list5.append((float(encoded_attach_rate_list[(i%672)])/6.)+(-1)*random.uniform(0.05,0.09))
    else:
        assign_array.append((float(encoded_attach_rate_list[(i%672)])/6.)+random.uniform(0.05,0.09))
        plotting_list5.append((float(encoded_attach_rate_list[(i%672)])/6.)+random.uniform(0.05,0.09))
        
            
    #assign it  to x-attributes & y-attrivutes
    test_x_attributes[i][place_to_insert_data] = assign_array
    
    
    
    

for i in range(672):
    for k in range(8):
        test_y_attributes[i][k] = float(test_x_attributes[(i+1+k) %  672][(i+1+k)%12][2]   ) 
        
        
for i in range(672):
    place_to_insert_data = i%12
    prob_var = random.random()
    if(prob_var>0.95):
        test_x_attributes[i][place_to_insert_data][2] = random.uniform(7,10)/6.
        pass
     
#GENERATE DATA FOR VALIDATING ONLINE LEARNING
     
for i in range(672):
    place_to_insert_data = i%12
    #Create temp array
    assign_array = []
    day_number = i%7
    time_number= i%96
    assign_array.append(float(day_number)/6.)
    assign_array.append(float(time_number)/96.)
    assign_array.append(float(online_attach_rate_list[i])/9.)
    #assign it  to x-attributes & y-attrivutes
    online_x_attributes[i][place_to_insert_data] = assign_array
    online_y_attributes[i] = float(online_attach_rate_list[(i+1)%672])/9.
    #just for plotting
    plotting_list6.append(float(online_attach_rate_list[i])/9.)
'''
#GENERATE TEST DATA FOR ONLINE LEARNING
    
for i in range(672):
    place_to_insert_data = i%12
    #Create temp array
    assign_array = []
    day_number = i%7
    time_number= i%96
    assign_array.append(float(day_number)/6.)
    assign_array.append(float(time_number)/96.)
    
    plus_minus_prob = random.random()
    anomaly_prob = random.random()    
    if(plus_minus_prob<0.65):
        assign_array.append((float(online_attach_rate_list[(i%672)])/9.)+(-1)*random.uniform(0.05,0.09))
        plotting_list7.append((float(online_attach_rate_list[(i%672)])/9.)+(-1)*random.uniform(0.05,0.09))
    else:
        assign_array.append((float(online_attach_rate_list[(i%672)])/9.)+random.uniform(0.05,0.09))
        plotting_list7.append((float(online_attach_rate_list[(i%672)])/9.)+random.uniform(0.05,0.09))
        
            
    #assign it  to x-attributes & y-attributes
    test_online_x_attributes[i][place_to_insert_data] = assign_array
    
    
    
    

for i in range(672):
    for k in range(8):
        test_online_y_attributes[i][k] = test_online_x_attributes[(i+k+1)%672][(i+k+1)%24][1]
        
        
for i in range(672):
    place_to_insert_data = i%12
    prob_var = random.random()
    if(prob_var>0.95):
        test_online_x_attributes[i][place_to_insert_data][2] = random.uniform(10,12)/9.
        pass    

    
plt.subplot(7, 1, 1)
plt.plot(np.array(plotting_list1))
plt.subplot(7, 1, 2)
plt.plot(np.array(plotting_list2))
plt.subplot(7, 1, 3)
plt.plot(np.array(plotting_list3))
plt.subplot(7, 1, 4)
plt.plot(np.array(plotting_list4))
plt.subplot(7, 1, 5)
plt.plot(np.array(plotting_list5))
plt.subplot(7, 1, 6)
plt.plot(np.array(plotting_list6))
plt.subplot(7, 1, 7)
plt.plot(np.array(plotting_list7))
plt.show()

#
#pickle.dump(x_attributes,open('x_att.p','wb'))
#pickle.dump(y_attributes,open('y_att.p','wb'))
#
#pickle.dump(test_x_attributes,open('test_x_att.p','wb'))
#pickle.dump(test_y_attributes,open('test_y_att.p','wb'))


#pickle.dump(online_x_attributes, open('online_x_att.p','wb'))
#pickle.dump(online_y_attributes, open('online_y_att.p','wb'))

pickle.dump(test_online_x_attributes, open('test_online_x_att.p','wb'))
pickle.dump(test_online_y_attributes, open('test_online_y_att.p','wb'))