from TelematicsHelper import get_data
from TelematicsHelper import douglas_pecker
from numpy import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pylab

data = get_data(1, 1)
print data.shape

pylab.plot(data[:, 0], data[:,1])
pylab.show()


ind = douglas_pecker(data, 1.0)
