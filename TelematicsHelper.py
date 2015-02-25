# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import numpy.linalg as LA
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math
from numpy import genfromtxt
from IPython.display import display
import pandas as pds
import os
from scipy import interpolate
import pylab as P
import simple_hist_features as shf
from scipy import interp
from sklearn import svm, datasets, linear_model
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, KFold
import time





from tempfile import NamedTemporaryFile
VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    
    return VIDEO_TAG.format(anim._encoded_video)

from IPython.display import HTML

def display_animation(anim):
    plt.close(anim._fig)
    return(HTML(anim_to_html(anim)))

def get_driver_ids():
    dirs = os.listdir(os.environ['TELEMATICS'])
    return list(map(int,sorted(dirs[1:])))

#make it global to save time
driver_ids = get_driver_ids()

def get_data(i,j):
    i = driver_ids[i-1]
    pre_path = os.environ['TELEMATICS']
    path = "{0}{1}/{2}.csv".format(pre_path,i,j)
    return genfromtxt(path, delimiter=',', skip_header = 1)



def calibrate_scores(probas):
    even_spaced = np.linspace(0,1,200)
    sorted_orig_indices = sorted(range(200), key= lambda(x): probas[x])
    sorted_indices = sorted(range(200), key = lambda(x): sorted_orig_indices[x])
    return even_spaced[sorted_indices]


def score_driver(d_id, n_folds=2):
    X, y = shf.get_even_training_data(d_id)
    probas = np.zeros(200)
    cv = StratifiedKFold(y, n_folds, shuffle=True)
    random_state = np.random.RandomState(0)
    classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)
    
    for i, (train, test) in enumerate(cv):
        then = time.clock()
        fft = classifier.fit(X[train], y[train])
        print(time.clock()-then)
        positive_test = test[test<200]
        probas[positive_test] = fft.predict_proba(X[positive_test])
    return calibrate_scores(probas)

# BINS
def get_bins():
    return 10

def average_features():
    
    BINS = get_bins()
    num_routes = 20
    routes = np.random.choice(200,num_routes,replace=False)
    dr_ids = len(get_driver_ids())
    tmp_data = get_data(1,1)
    tmp_features = shf.get_features(tmp_data,True)
    NUM_FEAT = np.size(tmp_features)
    total = np.zeros(NUM_FEAT)
    for i in range(1,1+dr_ids):
        for route in routes:
            total += shf.get_features(get_data(i,route+1))/num_routes
    return total/dr_ids
            
        

#incomplete and bad approach    
'''
def score_driver(d_id,n_folds = 10):
    probas = np.zeros(200)
    
    cv1 = KFold(200, n_folds, shuffle=True)
    cv0 = KFold(200, n_folds, shuffle=True)

    X1 = X[:200]
    y1 = np.ones(200)
    X0 = X[200:]
    y0 = np.zeros(200)
    random_state = np.random.RandomState(0)

    classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)
    
    for i, ((train1,test1),(train0,test0)) in enumerate(zip(cv1,cv0)):
        Xtrain = np.concatenate([X1[train1],X0[train0]])
        ytrain = np.concatenate([np.ones(len(train1)),np.zeros(len(train0))])
        perm = np.random.permutation(len(ytrain))
        Xtrain = Xtrain[perm]
        ytrain = ytrain[perm]
        
        fft = classifier.fit(Xtrain, ytrain)
        probas[test1] = fft.predict_proba(X1[test1])
        
        
    return calibrate_scores(probas)'''

def draw_trip(i,j,epsilon=0,X = None):
    #Use X for overriding trip data and provide your own data.
    draw_every = 10
    speed_up_factor = 10
    if X is None:
        X = get_data(i,j)
    m,_ = X.shape
    grid = list(range(0,m,draw_every))
    minx,maxx,miny,maxy = (np.min(X[:,0]) , np.max(X[:,0]) , np.min(X[:,1]), np.max(X[:,1]))
    
    X_doug, times = douglas_pecker(X,epsilon)  # testing how much smoothing we can use and still have sufficient info on curv
    curv = curvature(X_doug,times)
        

    curv_t = (times[1],times[-2])
    curv_interp = interpolate.interp1d(times[1:-1],curv,kind='linear')
    
    
    cmax = np.amax(np.abs(curv))    
    
    fig = plt.figure()
    sub = fig.add_subplot(121,xlim=(minx, maxx), ylim=(miny, maxy))
    sub2 = fig.add_subplot(122,xlim=curv_t, ylim=(-2*cmax, 2*cmax))
    
    tracker, = sub.plot([], [], 'ro')
    
    tracker2, = sub2.plot([],[],'ro')
    
    
    def init_func():
        line, = sub.plot(X[:,0],X[:,1], lw=2)
        line2, = sub2.plot(times[1:-1],curv, lw=2)
        
        return line, line2
    
    
    def run(t_from_grid):
        # update the data
        path = X[t_from_grid,:]
        t = float(np.clip(t_from_grid,*curv_t))
        tracker.set_data(path[0], path[1])
        tracker2.set_data(t,curv_interp(t))
        return tracker, tracker2
    
    interval = 1000*draw_every/speed_up_factor
    
    ani = animation.FuncAnimation(fig, run, grid, blit=True, interval=interval,
        repeat=False, init_func = init_func)
    #return ani
    display(display_animation(ani))
    

def draw_vt(i,j):
    data = get_data(i,j)
    vel = velocity(data);
    fig, ax = plt.subplots()
    ax.plot(range(1,1+len(vel)),vel)


def rescale_route(data,step_length=10):
    #rescale_route(data,step_length=10):
    #we sample points from the route given by data linearly interpolated so that each
    #sampling point is at distance step_length from the previous, except for the last point.

    index = 1
    prev = data[0]
    result = [prev]
    inside = prev
    times = [0]
    T = 0
    remaining_of_sec = 1.0
    
    while index < len(data):
        while index < len(data) and LA.norm(data[index]-prev)<=step_length:
            inside = data[index]
            T = index
            remaining_of_sec = 1.0
            index += 1
        if index == len(data):
            new_point = inside
            times.append(index-1)
        else:
            diff1 = inside - prev
            diff2 = data[index] - inside
            diff2sq = np.dot(diff2,diff2)
            c_proj = np.dot( diff1, diff2 ) / diff2sq
            t = -c_proj + math.sqrt(c_proj**2 + (step_length**2 - np.dot(diff1,diff1))/diff2sq)
            T += t*remaining_of_sec
            remaining_of_sec *= (1-t)
            times.append(T)
            new_point = inside + t*diff2
            prev = new_point
            inside = new_point
            
        result.append(new_point)
    return times, np.array(result)
    ####
    '''
    total_length = 0
    n,_ = data.shape
    for i in range(n-1):
        total_length+= LA.norm(data[i+1,:]-data[i,:])
    
    length_to_go_total = total_length
    total_steps = total_length // step_length

    dat = np.zeros((total_steps+1,2))
    times = np.zeros(total_steps+1)
    dat[0,:] = data[0,:]

    # this is a point that tracks along the curve until it has travelled step_length
    point = data[0,:]
    points_added = 1
    next_index = 1
    next_point = data[next_index,:]

    while length_to_go_total >= step_length:
        length_to_go = step_length
        to_next_point = LA.norm(next_point-point)
        while to_next_point <= length_to_go:
            length_to_go -= to_next_point
            point = next_point
            next_index+=1
            next_point = data[next_index,:]
            to_next_point = LA.norm(next_point-point)
        proportion = length_to_go/to_next_point
        point = (1-proportion)*point + proportion*next_point
        time = next_index - 1 + proportion;
        dat[points_added,:] = point
        times[points_added] = time
        points_added+=1
        length_to_go_total -= step_length
    return times, dat'''


    

def point_line_dist(p1,p2,x):
    unit_normal = p2-p1
    unit_normal[0], unit_normal[1] = (-unit_normal[1], unit_normal[0])
    unit_normal /= LA.norm(unit_normal)
    return abs(np.dot(x-p1, unit_normal))

def douglas_pecker(data, epsilon):
    # times returned are the indices chosen for the reduced trajectory
    #// Find the point with the maximum distance
    dmax = 0
    index = 1
    length = np.size(data,0)
    for i in range(1,length-1):
        d = point_line_dist(data[0], data[-1], data[i]) 
        if d > dmax :
            index = i
            dmax = d
    #// If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        #// Recursive call
        recResults1,times1 = douglas_pecker(data[:index+1], epsilon)
        recResults2,times2 = douglas_pecker(data[index:], epsilon)
        times = np.concatenate((times1[:-1],times2+index))
 
        #// Build the result list
        result = np.concatenate((recResults1[:-1],recResults2), axis = 0)
    else:
        result = data[[0,-1]]
        times = np.array([0,len(data)-1])
    #// Return the result
    return result, times

def test_pecker(i,j,epsilon,data=None):
    if data is None:
        data = get_data(i,j)
    data, _ = douglas_pecker(data,epsilon)
    plt.scatter(*data.T)
    

def draw_spline_curvature(data,times):
    tck,_ = interpolate.splprep(data.T,u=times)
    thick_times = np.zeros(10*len(times) - 9)
    for i in range(len(times)-1):
        thick_times[10*i:10*i+10] = np.linspace(times[i],times[i+1],10)
    thick_times[-1] = times[-1]
    interpolated_data = interpolate.splev(thick_times,tck)
    tangent = np.array(interpolate.splev(times,tck,der=1))
    tangent_norm = LA.norm(tangent,axis=0)
    second_derivative = interpolate.splev(times,tck,der=2)
    
    curvature = np.abs(tangent[0]*second_derivative[1] - tangent[1]*second_derivative[0])/tangent_norm**3
    plt.figure(1)
    plt.subplot(121)
    plt.plot(data[:,0], data[:,1], 'rx', interpolated_data[0] , interpolated_data[1] , 'k-')
    plt.subplot(122)
    plt.plot(times, curvature, 'rx')
    plt.axis([times[0],times[-1],np.amin(curvature),np.amax(curvature)])
    

#code for drawing tangent arrows    
'''    
for i,d in enumerate(data):
    P.arrow( d[0], d[1], second_derivative[0,i], second_derivative[1,i], fc="k", ec="k",
    head_width=0.05, head_length=0.1 )'''
    
    
def draw_curvature(i,j,epsilon=5,data=None):
    if data is None:
        data = get_data(i,j)
    data, times = douglas_pecker(data,epsilon)
    #curv = pds.rolling_mean(curvature(data,times),10)
    curv = curvature(data,times)
    fig = plt.figure()
    cmax = np.amax(np.abs(curv))
    sub=fig.add_subplot(111,xlim=(times[1],times[-2]) , ylim=(-2*cmax,2*cmax))
    sub.plot(times[1:-1],curv)
    

    
def curvature(data,times=None):
    n = np.size(data,0)
    if times is None:
        times = np.arange(n)
    curv = np.zeros(n - 2)
    diff = data[1] - data[0]
    norm = LA.norm(diff)
    
    prev_diff = None
    prev_norm = None
    
    for i in range(1,n-1):
        prev_diff = diff
        prev_norm = norm
        diff = (data[i+1] - data[i])/(times[i+1] - times[i])
        norm = LA.norm(diff)
        ddiff = 2*(diff - prev_diff)/(times[i+1] - times[i-1])
        curv[i-1] = (diff[0]*ddiff[1] - diff[1]*ddiff[0])/norm**3
    return curv
    

def velocity(data):
    #compute velocity as norm of difference of consecutive coordinates.
    # we can take differences over a window by setting window > 1.
    window = 1
    n=np.size(data,0)
    return 3.6*LA.norm(data[window:,:] - data[:n-window,:],axis = 1)


def accel(data, vel=None):
    #compute tangential accelearation
    if (vel is None):
        vel = velocity(data)
    return np.diff(vel)

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = np.cross(v1, v2)
    return np.arctan2(sinang, cosang)

def angles(data):
    diff = np.diff(data, axis=0)
    return np.array([py_ang(diff[i,:], diff[i+1,:]) for i in xrange(diff.shape[0]-1)])
    

