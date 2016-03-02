# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:15:12 2016

@author: suraj
"""


#Make a list of 6064 numbers between 4-6
import random
import numpy as np
import pickle
import scipy

attachRateList = []
for i in range(100):
    probabilityVariable = random.random()
    if(probabilityVariable< 0.9):
        attachRateList.append(random.uniform(4,6))
    else:
        attachRateList.append(random.uniform(4,6))

attachRateList = np.array(attachRateList)
encoded_attach_rate_list = np.fft.fft(attachRateList)
inverse_attach_rate_list = np.fft.ifft(encoded_attach_rate_list)
import matplotlib.pyplot as plt

encoded_days = np.fft.fft(np.array(range(6)))
plt.subplot(3,1,1)
plt.plot(range(len(attachRateList)),attachRateList)
plt.subplot(3,1,2)
plt.plot(np.fft.fftfreq(np.array(encoded_attach_rate_list).shape[-1]),encoded_attach_rate_list.imag)
plt.subplot(3,1,3)
plt.plot(np.fft.fftfreq(np.array(encoded_attach_rate_list).shape[-1]),encoded_attach_rate_list.real)

#print encoded_attach_rate_list
#plt.subplot(3,1,3)
#plt.plot(np.fft.fftfreq(encoded_days.shape[-1]),encoded_days.imag)
plt.show()



a = np.fft.fftfreq(np.array(encoded_attach_rate_list).shape[-1])

b = encoded_attach_rate_list.imag

for i in range(len(a)):
    print a[i],b[i]

#print inverse_attach_rate_list

ap = attachRateList
bp = inverse_attach_rate_list.real

for j in range(len(ap)):
    print ap[j] ,bp[j]