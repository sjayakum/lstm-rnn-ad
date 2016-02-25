import pickle
import random
import numpy
x = []


for i in range(4):
    temp1 = []
    for i in range(288):
        temp2  = []
        temp2.append(0)
        temp2.append(random.uniform(4,6))
        temp1.append(temp2)
    x.append(temp1)

newx = numpy.array(x)

print newx.shape
print newx

y = []

for i in range(4):
    temp3 = []
    for j in range(287):
        temp3.append([newx[i][j+1][1]])
    temp3.append([newx[i][j][1]])
    y.append(temp3)

newy = numpy.array(y)

print newy.shape
print newy

print "*****************************"

print newx[0][:3]
print newy[0][:3]


pickle.dump(newx,open('../mondayXwind.p','wb'))
pickle.dump(newy,open('../mondayYwind.p','wb'))








'''
************* TEST DATA CREATION ****************
'''
xyz = []


for i in range(4):
    temp1 = []
    for i in range(288):
        temp2  = []
        temp2.append(0)
        probVar = random.random()
        if(probVar > 0.7):
            temp2.append(random.uniform(6,10))
        else:
            temp2.append(random.uniform(4,6))
        temp1.append(temp2)
    xyz.append(temp1)

newxyz = numpy.array(xyz)


print newxyz
y = []

for i in range(4):
    temp3 = []
    for j in range(287):
        temp3.append([newxyz[i][j+1][1]])
    temp3.append([newxyz[i][j][1]])
    y.append(temp3)

newy = numpy.array(y)

print newy

pickle.dump(newxyz,open('../tesmondayXwind.p','wb'))
pickle.dump(newy,open('../tesmondayYwind.p','wb'))
