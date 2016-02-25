
import pickle
import numpy as np

def getAnomalyCount(filename):
    x = pickle.load(open(filename))
    anacount = 0
    for i in range(4):
    	for j in range(288):
	        if x[i][j][1] >6:
	            anacount = anacount + 1
    return anacount

print getAnomalyCount('tesmondayX.p')
