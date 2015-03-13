from Driver import Driver
from random import sample, seed
import logging
import os

import StartRegression as start_reg

import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

folder = os.environ['TELEMATICS']
foldername = folder
referencenum = 10

global NUM_REF_DRIVERS

folders = [os.path.join(foldername, f) for f in os.listdir(foldername) if os.path.isdir(os.path.join(foldername, f))]
referencefolders = [folders[i] for i in sorted(sample(xrange(len(folders)), referencenum))]
logging.debug(referencefolders)
referencedrivers = []
    
for referencefolder in referencefolders:
    referencedrivers.append(Driver(referencefolder))
    
start_reg.generatedata(referencedrivers)

pred = np.array([])
start_reg.REFERENCE_DATA["num_ref_drivers"] = 4    
start_reg.classify("/Users/zhelezov/coding/octave/telematics/Telematics/drivers/1048", toKaggle = True)
# cv_1, cv_2, pr = start_reg.grid_search("/Users/zhelezov/coding/octave/telematics/Telematics/drivers/1048")
#    pred = np.hstack((pred, pr))
#    logging.info("Round 1: %.3f, Round 2: %.3f", cv_1, cv_2)



#pred = np.reshape(pred, (200, -1))
#logger.info("%s", pred)

#logger.info("%s", np.array([np.mean(pred, axis = 1), np.std(pred, axis=1)]))



