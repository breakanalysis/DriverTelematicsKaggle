# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import TelematicsHelper as th
import matplotlib.pyplot as plt
import numpy as np

def get_features(data, concat=False, plot=False):
    #.shape
    len_t = np.shape(data)[0]

    times,X = th.rescale_route(data)
    curv = [] #th.curvature(data)
    vel = th.velocity(data)
    acc = th.accel(data, vel)
    angles = th.angles(data)*180.0/np.pi

    
    
    #just put everything in a huge multidimensional histogram
    v_acc_ang_h, edges = np.histogramdd([vel[:-1], acc, angles], range=([0, 160], [-20, 20], [-180, 180]), bins=20)
    
    if (concat):
        return np.ravel(v_acc_ang_h)
    
    return v_acc_ang_h, edges#vel_h, acc_h, curv_h, angles_h


def get_training_data(driver_1_id, drivers_0_ids, r_ids=np.arange(1,201)):
## generate one big training data by labeled with ones for driver_1 paths 
## versus all other drivers routes, labeled. 
    r_num = np.size(r_ids)
    d_num = np.size(drivers_0_ids)
    ## ones for the first driver and zeroes for the others
    labels = np.concatenate([np.full(r_num, 1, dtype=int), np.full(r_num*d_num, 0, dtype=int)])
    
    sample_data = th.get_data(driver_1_id, 1)
    sample_feat = get_features(sample_data, True)
    
    X = np.zeros(((d_num+1)*r_num, np.size(sample_feat)))
    i = 0 #index of routes
    for r_id in r_ids:
        d1_data = th.get_data(driver_1_id, r_id)
        X[i, :] = get_features(d1_data, True)
        j = 1 #index of drivers
        for d_id in drivers_0_ids:
            d0_data = th.get_data(d_id, r_id)
            X[j*r_num + i, :] = get_features(d0_data, True)
            j += 1
        i += 1    
            
        
    return X, labels
       

# <codecell>

data = th.get_data(1,2)
hh = get_features(data, True, False)

print np.size(hh)
print np.size(data)

# <codecell>

data = th.get_data(1,10)
feat = get_features(data, True, True)


# <codecell>

print np.diff(data, axis=0)

# <codecell>

cosang = np.dot([ 0., 1.], [ 1.,  0.])
sinang = np.cross([ 0.,  1.], [ 1.,  0.])
print np.arctan2(sinang, cosang)

cosang = np.dot([ 1.,  0.], [ 0.,  1.])
sinang = np.cross([ 1.,  0.], [ 0.,  1.])
print np.arctan2(sinang, cosang)


# <codecell>

labels = np.concatenate([np.full(5, 1,dtype=int), np.full(5,0,dtype=int)])

# <codecell>

print labels

# <codecell>


# <codecell>

print np.shape(th.get_driver_ids())

# <codecell>

x = th.get_driver_ids()
y = np.delete(x, 1)

# <codecell>

print np.shape(y)

# <codecell>

print np.shape(x)

# <codecell>

print np.arange(1,10)

# <codecell>


