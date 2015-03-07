from Driver import Driver
from random import sample, seed
import logging
import os

import StartRegression as reg_slow

import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

folder = os.environ['TELEMATICS']
foldername = folder
referencenum = 5


seed(43)
folders = [os.path.join(foldername, f) for f in os.listdir(foldername) if os.path.isdir(os.path.join(foldername, f))]
referencefolders = [folders[i] for i in sorted(sample(xrange(len(folders)), referencenum))]
logging.debug(referencefolders)
referencedrivers = []
    
for referencefolder in referencefolders:
    referencedrivers.append(Driver(referencefolder))
    
reg_slow.generatedata(referencedrivers)
    
reg_slow.grid_search("/Users/zhelezov/coding/octave/telematics/Telematics/drivers/1048")


