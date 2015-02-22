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
    curv = th.curvature(data)
    vel = th.velocity(data)
    acc = th.accel(data, vel)
    angles = th.angles(data)*180.0/np.pi

    curv = curv[np.isfinite(curv)]

    #acc_bins = np.concatenate([np.linspace(-20, -1, 20, endpoint=False), 
    #                           np.linspace(-1,1,20, endpoint=False), np.linspace(1,20,20)])

    #curv_bins = np.concatenate([np.linspace(0,3,40)])

    #v_bins = np.concatenate([np.linspace(0,5,20, endpoint=False), np.linspace(20,90,30, endpoint=False),
    #                         np.linspace(90,150,5)])

    
    curv_bins = np.exp(np.linspace(-3, 1, 40))
    v_bins = np.exp(np.linspace(-1, 5, 30))
    acc_bins = np.concatenate([-np.exp(np.linspace(3,-8, 20)),np.exp(np.linspace(-8,3, 20))]) 
    ang_bins = np.concatenate([np.linspace(-25,-5,5, endpoint=False), np.linspace(-5,5,20, endpoint=False),
                             np.linspace(5,25,5)])

    
    if (plot):
        plt.figure(1)
        plt.subplot(141)
        plt.hist(vel, bins=v_bins, normed=True)
        plt.subplot(142)
        plt.hist(acc, bins=acc_bins, normed=True)
        plt.subplot(143)
        plt.hist(curv, bins=curv_bins, normed=True)
        plt.subplot(144)
        plt.hist(angles, bins=ang_bins, normed=True)

    curv_h, curv_e = np.histogram(curv, bins=curv_bins, density=True)
    angles_h, angles_e = np.histogram(angles, bins=ang_bins, density=True)
    vel_h, vell_e = np.histogram(vel, bins=v_bins, density=True)
    acc_h, acc_e = np.histogram(acc, bins=acc_bins, density=True)
    
    if (concat):
        return np.concatenate([vel_h, acc_h, curv_h, angles_h])
    
    return vel_h, acc_h, curv_h, angles_h


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

d_0 = np.random.choice(np.arange(2, 2735), 20, replace=False)
r_indx = np.random.choice(np.arange(1, 201), 20, replace=False)
X, labels = get_training_data(1, d_0, r_indx)

print np.shape(X), np.shape(labels)

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


