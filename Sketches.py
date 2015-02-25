# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%run TelematicsHelper
import numpy as np
import numpy.linalg as LA
import math
import matplotlib.pyplot as plt

i,j = (1,1)
epsilon = 2
#test_pecker(i,j,epsilon)
data = get_data(i,j)
data, times = douglas_pecker(data,80)

times = list(range(np.size(data,0)))

#data = np.array([[np.cos(t),np.sin(t)] for t in np.linspace(0,math.pi,20)])
#times = list(range(20))

#data.shape
#np.size(times)

draw_spline_curvature(data,times)
print accel(data)

# <codecell>


# <codecell>

#draw_curvature(52,13,0,np.array([[np.cos(t),np.sin(t)] for t in np.linspace(0,2*math.pi,150)]))
#test_pecker(-1,1,0.001,np.array([[np.cos(t),np.sin(t)] for t in np.linspace(0,2*math.pi,150)]))

data = np.array([[np.cos(t),np.sin(t)] for t in np.linspace(0,2*math.pi,150)])
datap,times = douglas_pecker(data,0)

draw_curvature(1,1,0)

# <codecell>

X = get_data(34,46)
X.shape

times,X = rescale_route(X)

fig,ax = plt.subplots()
acc =accel(X)
plt.plot(list(range(1,np.size(acc)+1)), acc)

#real speeds
#draw_trip(44,46,0)
#constant speed
draw_trip(0,0,1,X)

# <codecell>

data=np.array([[1.1,2.2],[3.5, 4.56], [6.456,3.45],[4.45,5.45], [1.45, 12.5]])
print np.diff(data, axis=0)
print angles(data)

# <codecell>

%run TelematicsHelper

a= score_driver(1)

# <codecell>

import time
import simple_hist_features as shf
then = time.clock()
X, y = shf.get_even_training_data(4)
tm = time.clock() - then
print(tm)
X.shape
    

# <codecell>

from sys import getsizeof
getsizeof(X)
X.nbytes
X.shape

# <codecell>


