# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import TelematicsHelper as th
import matplotlib.pyplot as plt
import numpy as np

BINS = 20
FEATURES_NUM = BINS**3

def get_features(data, concat=False, plot=False):
    #.shape
    len_t = np.shape(data)[0]

    times,X = th.rescale_route(data)
    curv = [] #th.curvature(data)
    vel = th.velocity(data)
    acc = th.accel(data, vel)
    angles = th.angles(data)*180.0/np.pi

    
    
    #just put everything in a huge multidimensional histogram
    v_acc_ang_h, edges = np.histogramdd([vel[:-1], acc, angles], 
        range=([0, 160], [-20, 20], [-180, 180]), bins=BINS, normed=True)
    
    if (concat):
        return np.ravel(v_acc_ang_h)
    
    return v_acc_ang_h, edges#vel_h, acc_h, curv_h, angles_h


def get_all_driver_features(driver_id):
    X = np.zeros((200, FEATURES_NUM))
    for i in range(0, 200):
        d1_data = th.get_data(driver_id, i+1)
        X[i, :] = get_features(d1_data, True)
    
    return X


def get_even_training_data(driver_id, n_routes=200, all_drivers=np.arange(1, 2735), randomize=True, weight=1):
    rest_d = np.delete(all_drivers, driver_id)
    size_0 = int(weight*n_routes)
    size_1 = n_routes
    routes_1 = np.random.choice(200, n_routes, replace=False)


    labels = np.concatenate([np.full(size_1, 1, dtype=int), np.full(size_0, 0, dtype=int)])
    drivers_0 = np.random.choice(rest_d, size_0, replace=False)  #prevent duplicates
    routes_0 = np.random.choice(200, size_0)
    
    
    X = np.zeros((size_1 + size_0, FEATURES_NUM))
    
    for i in range(0, size_1):
        d1_data = th.get_data(driver_id, routes_1[i]+1)
        X[i, :] = get_features(d1_data, True)
    
    shift = size_1
    
    for i in range(0, size_0):
        d0_data = th.get_data(drivers_0[i], routes_0[i] + 1)
        X[shift + i, :] = get_features(d0_data, True)
    
    return X, labels
    
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
       
