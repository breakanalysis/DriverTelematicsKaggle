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
    dirs = os.environ['TELEMATICS']
    return list(map(int,sorted(os.listdir(dirs))))




def get_data(i,j):
    i = get_driver_ids()[i-1]
    pre_path = os.environ['TELEMATICS']
    path = "{0}{1}/{2}.csv".format(pre_path,i,j)
    return genfromtxt(path, delimiter=',', skip_header = 1)

def draw_trip(i,j):
    i = get_driver_ids()[i-1]
    draw_every = 10
    speed_up_factor = 10
    X = get_data(i,j)
    m,_ = X.shape
    X_sparsed = X[range(0,m,draw_every),:]
    fig, ax = plt.subplots()
    tracker, = ax.plot([], [], 'ro')
    minx,maxx,miny,maxy = (np.min(X[:,0]) , np.max(X[:,0]) , np.min(X[:,1]), np.max(X[:,1]))
    ax.set_ylim(miny, maxy)
    ax.set_xlim(minx, maxx)
    
    def init_func():
        line, = ax.plot(X[:,0],X[:,1], lw=2)
        return line,
        
    def run(data):
        # update the data

        tracker.set_data(data[0], data[1])

        return tracker,

    interval = 1000*draw_every/speed_up_factor
    
    ani = animation.FuncAnimation(fig, run, X_sparsed, blit=True, interval=interval,
        repeat=False, init_func = init_func)
    #return ani
    display(display_animation(ani))


def draw_vt(i,j):
    data = get_data(i,j)
    vel = velocity(data);
    fig, ax = plt.subplots()
    ax.plot(range(1,1+len(vel)),vel)


def rescale_route(data,step_length=10):
    #gives a new set of coordinates with the same number of measurements,
    #but each step has now the same length. to obtain the coordinates, we
    #let a particle traverse the route (linearly interpolated) with the velocity  step_length m/s,
    #and we record its position every second.


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
    return times, dat

def mydot(v1,v2):
    return sum((x*y for x,y in zip(v1,v2)))


def velocity_curvature_avg(data):
    #compute velocity and curvature (1/radius). velocity is computed looking ahead by window.
    #curvature is also computed by computing difference vector over the window and
    #approximating the length of the osculating circular arc by the length of the straight
    #line segment 1/window*(P_(t+window) - P_(t)). angle is computed between
    #two such consecutive line segments. then we use s = r * angle.  

    n=np.size(data,0)
    window = 1
    vel = np.zeros((n-window,1),dtype=float)
    curvature = np.zeros((n-window,1),dtype=float)
    dir_right = np.zeros((n-window,1),dtype=int)
    vel[0] = LA.norm(data[window,:] - data[0,:])
    for i in range(1,n-window):
        difff = data[i+window,:] - data[i,:]
        prev_difff = data[i+window-1,:] - data[i-1,:]
        difff_ccw90 = [-difff[1],difff[0]]
        vel[i] = LA.norm(difff)
        if (vel[i]*vel[i-1] == 0):
            curvature[i] = np.nan
        else:
            # have to avoid nummerical error leading to acos(x),  where x>1
            angle = math.acos(0.99999*mydot(difff,prev_difff)/(vel[i-1]*vel[i]))
            curvature[i] = window*angle/vel[i]
            dir_right[i] = np.sign(mydot(prev_difff,difff_ccw90))
    dat = np.concatenate((3.6*vel[1:]/window,3.6*curvature[1:],dir_right[1:]),axis = 1)
    return dat

def curvature(data,out_times):
    #compute curvature (1/radius). we first apply rescale_route to get rid of numerical issues
    #arising from low velocities.
    #curvature is computed by computing difference vector over the window and
    #approximating the length of the osculating circular arc by the length of the straight
    #line segment 1/window*(P_(t+window) - P_(t)). angle is computed between
    #two such consecutive line segments. then we use s = r * angle.  
    step_length = 10
    #times, data = rescale_route(data,step_length)
    times = out_times
    n=np.size(data,0)
    window = 1
    #vel = np.zeros((n-window,1),dtype=float)
    curvature = np.zeros((n-window,1),dtype=float)
    #vel[0] = LA.norm(data[window,:] - data[0,:])
    for i in range(1,n-window):
        difff = data[i+window,:] - data[i,:]
        prev_difff = data[i+window-1,:] - data[i-1,:]
        difff_ccw90 = [-difff[1],difff[0]]
        vel = LA.norm(difff)
        prev_vel = LA.norm(prev_difff)
        if (vel*prev_vel == 0):
            curvature[i] = np.nan
        else:
            # have to avoid nummerical error leading to acos(x),  where x>1
            angle = math.acos(0.99999*mydot(difff,prev_difff)/(vel*prev_vel))
            curvature[i] = np.minimum(window*angle/vel,math.pi*window/step_length)
            curvature[i] *= np.sign(mydot(prev_difff,difff_ccw90))
    curv = curvature[1:]
    print(np.amin(curv))
    
    times = times[1:-window]
    last_ind = n - window -2
    result = np.zeros_like(out_times,dtype=float)
    i = 0
    for j,t in enumerate(out_times):
        while i < last_ind and t > times[i]:
            i+=1
        result[j] = curv[i]
    return result

def point_line_dist(p1,p2,x):
    unit_normal = p2-p1
    unit_normal[0], unit_normal[1] = (-unit_normal[1], unit_normal[0])
    unit_normal /= LA.norm(unit_normal)
    return abs(np.dot(x-p1, unit_normal))

def douglas_pecker(data, epsilon):
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
        recResults1 = douglas_pecker(data[:index+1], epsilon)
        recResults2 = douglas_pecker(data[index:], epsilon)

 
        #// Build the result list
        result = np.concatenate((recResults1[:-1],recResults2), axis = 0)
    else:
        result = data[[0,-1]]
    #// Return the result
    return result

def test_pecker(i,j,epsilon):
    i = get_driver_ids()[i]
    data = get_data(i,j)
    data = douglas_pecker(data,epsilon)
    plt.scatter(*data.T)

def fit_spline(data):
    #TODO
    return None

def draw_curvature(i,j):
    data = get_data(i,j)
    out_times = range(np.size(data,0)+1)
    curv = pds.rolling_mean(curvature(data,out_times),120)
    print("length" + str(np.size(curv)))
    fig, ax = plt.subplots()
    ax.plot(range(1,1+len(curv)),curv)
    sorted_curv = sorted(curv)
    for i in curv:
        None#print(i)
    n = len(sorted_curv)
    #low = sorted_curv[int(0.03*n)]
    #high = sorted_curv[int(0.97*n)]
    low = -0.0007
    high = 0.0007
    ax.axis([1,np.size(data,0),low,high])


def velocity(data):
    #compute velocity as norm of difference of consecutive coordinates.
    # we can take differences over a window by setting window > 1.
    window = 1
    n=np.size(data,0)
    return 3.6*LA.norm(data[window:,:] - data[:n-window,:],axis = 1)


# <codecell>

import numpy as np
import numpy.linalg as LA
import math
import matplotlib.pyplot as plt

i,j = (53,12)
epsilon = 2
test_pecker(i,j,epsilon)
data = get_data(i,j)
print(len(data))
print(len(douglas_pecker(data,epsilon)))



# <codecell>


