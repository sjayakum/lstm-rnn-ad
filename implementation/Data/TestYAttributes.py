import pickle
import random
import numpy as np

y = pickle.load(open('tesmondayY.p'))

Fnewy = []

for i in range(4):
    newy = []
    for j in range(288):
        if(4<y[i][j]<6):
		newy.append(y[i][j])
	else:
		newy.append(random.uniform(4,6))
    Fnewy.append(newy)
Fnewy = np.array(Fnewy)
pickle.dump(Fnewy,open('updated_tesmondayY.p','wb'))
