# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:15:12 2016

@author: suraj
"""



import random
import numpy as np
import pickle
import matplotlib.pyplot as plt


attachRateList = []

for i in range(3360):
    attachRateList.append(random.uniform(4,6))

attachRateList = np.array(attachRateList)
encoded_attach_rate_list = np.fft.fft(attachRateList)


day_number_list = [i%7 for i in range(3360)]
encoded_day_number_list = np.fft.fft(day_number_list)


time_number_list = [i%96 for i in range(3360)]
encoded_time_number_list = np.fft.fft(time_number_list)


final_list_x = np.array([[encoded_day_number_list.real[i],encoded_day_number_list.imag[i],encoded_time_number_list.real[i],encoded_time_number_list.imag[i],encoded_attach_rate_list.real[i],encoded_attach_rate_list.imag[i]] for i in range(3360)])

final_list_y = [ (encoded_attach_rate_list[i].real,encoded_attach_rate_list[i].imag) for i in range(len(encoded_attach_rate_list)) ]



pickle.dump(final_list_x,open('x_att.p','wb'))
pickle.dump(final_list_y,open('y_att.p','wb'))


