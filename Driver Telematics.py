# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%run TelematicsHelper.ipynb




#data = get_data(np.random.randint(2783)+1,np.random.randint(200)+1)

#draw_trip(1,2)
#draw_vt(13,52)
#draw_curvature(13,52)

data = np.array([[np.cos(t),np.sin(t)] for t in np.linspace(0,2*math.pi,100)])
out_times = range(np.size(data,0)+1)
curv = curvature(data,out_times)

#plt.scatter(data[:,0],data[:,1])
plt.plot(curv)

# <codecell>

import numpy as np
import math

int(0.97)

# <codecell>


# <codecell>


