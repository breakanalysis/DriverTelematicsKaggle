import os
from math import hypot
import TelematicsHelper as th
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import logging

def smooth(x, y, steps):
    """
    Returns moving average using steps samples to generate the new trace

    Input: x-coordinates and y-coordinates as lists as well as an integer to indicate the size of the window (in steps)
    Output: list for smoothed x-coordinates and y-coordinates
    """
    xnew = []
    ynew = []
    for i in xrange(steps, len(x)):
        xnew.append(sum(x[i-steps:i]) / float(steps))
        ynew.append(sum(y[i-steps:i]) / float(steps))
    return xnew, ynew


def distance(x0, y0, x1, y1):
    """Computes 2D euclidean distance"""
    return hypot((x1 - x0), (y1 - y0))


def velocities_and_distance_covered(x, y):
    """
    Returns velocities just using difference in distance between coordinates as well as accumulated distances

    Input: x-coordinates and y-coordinates as lists
    Output: list of velocities
    """
    v = []
    distancesum = 0.0
    for i in xrange(1, len(x)):
        dist = distance(x[i-1], y[i-1], x[i], y[i])
        v.append(dist)
        distancesum += dist
    return v, distancesum

BINS = 20

def hist_features(vel, acc, angles, concat=False, plot=False):

    #len_t = np.shape(data)[0]

    #times,X = th.rescale_route(data)
    #curv = [] 
    #vel = th.velocity(data)
    #acc = th.accel(data, vel)
    #angles = angles*180.0/np.pi
    
    #just put everything in a huge multidimensional histogram
    v_acc_ang_h, edges = np.histogramdd([vel[:-1], acc, angles], 
        range=([0, 90], [-20, 20], [-20, 20]), bins=BINS, normed=True)
    
    if (concat):
        return np.ravel(v_acc_ang_h)
    
    return v_acc_ang_h, edges



def get_trace_data(driver_id,trace_id):
    pre_path = os.environ['TELEMATICS']
    path = "{0}{1}/{2}.csv".format(pre_path, driver_id, trace_id)
    return genfromtxt(path, delimiter=',', skip_header = 1)


class Trace(object):
    """"
    Trace class reads a trace file and computes features of that trace.
    """

    def __init__(self, driver_id, trace_id, filtering=10):
        """Input: driver id and the trace id (from 1 to 200)"""
        self.__id = trace_id
        self.__driver_id = driver_id
        data = get_trace_data(driver_id, trace_id)
        self.data = data
        self.__xn, self.__yn = smooth(data[:,0], data[:,1], filtering)
        
        
        self.vel = th.velocity(data)
        self.acc = th.accel(data, self.vel)
        self.angles = th.angles(data)*180.0/np.pi

        self.maxspeed = max(self.vel)
        self.triplength = distance(data[0,0], data[0,1], data[-1,0], data[-1, 1])
        self.triptime = np.shape(data)[0]
        
        lin = np.linspace(10, 90, 20);

        self.v_perc = np.percentile(self.vel, lin)
        #logging.debug("Velocity percentiles of driver %d: %s", driver_id, self.v_perc)
        self.acc_perc = np.percentile(self.acc, lin)
        #logging.debug("Acc percentiles of driver %d: %s", driver_id, self.acc_perc)
        self.ang_perc = np.percentile(self.angles, lin)
        #logging.debug("Angles percentiles of driver %d: %s", driver_id, self.ang_perc)
        
        self.all_hist_features = hist_features(self.vel, self.acc, self.angles, True)


    @property
    def features(self):
        """Returns a list that comprises all computed features of this trace."""
        features = np.array([])
        features = np.append(features, self.triplength)
        features = np.append(features, self.triptime)
        features = np.append(features, self.maxspeed)

        features = np.append(features, self.v_perc)
        features = np.append(features, self.acc_perc)
        features = np.append(features, self.ang_perc)
        
        features = np.append(features, self.all_hist_features)
        return features

    def __str__(self):
        return "Trace {0} has this many positions: \n {1}".format(self.__id, self.triptime)

    @property
    def identifier(self):
        """Driver identifier is its filename"""
        return self.__id
