import pickle
import numpy as np
DataX  = []
examples = open('inet.csv').read().strip().split('\n')
daycounter = 0
attList = []
actualDofMonth = 0
for ex in examples:
    tempList = []
    y = ex.split(',')[1]
    actualDofMonth = ((daycounter) /24 %7)
    tempList.append(actualDofMonth)
    tempList.append(float(y)/150000)
    attList.append(tempList)
    daycounter = daycounter + 1



print len(attList)

superList = []
a1 = 0
pcounter =0
#import  pdb;pdb.set_trace()
for i in range(len(attList)/24):

    for j in range(24):

        finalList = [ [[0.,0.]]*24]
        finalList[0][j] = attList[a1:a1+24][j]
        # finalList[0][0:j] = attList[a1:a1+24][0:j]
        superList.append(finalList)
    a1 = a1+24


finalAns = np.array(superList)
#
# print finalAns.shape
# print finalAns[0]
# print finalAns[1]


pickle.dump(finalAns,open('xAttn.p','wb'))

yList = []

icounter = 0
#import pdb;pdb.set_trace()
for i in range(len(finalAns)-1):
    yList.append((attList[icounter + 1][1]))
    icounter = icounter + 1
yList.append(((attList[0][1])))

print attList[0][1]

print "***************************"
print attList[1][1]
print yList
yList = np.array(yList)
pickle.dump(yList,open('yAttn.p','wb'))

yFinal = 0
yFinal = []

for i in range(len(yList)):
    
    oheyList = [ [0]*24 ]
    oheyList[0][i%24] = yList[i]
    yFinal.append(oheyList)

print yFinal[0]
yFinal = np.array(yFinal)
print yFinal.shape
# pickle.dump(yFinal,open('yAttn.p','wb'))




#pickle.dump(yList,open('yAtt.p','wb'))
