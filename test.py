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
referencenum = 4

global NUM_REF_DRIVERS

folders = [os.path.join(foldername, f) for f in os.listdir(foldername) if os.path.isdir(os.path.join(foldername, f))]
referencefolders = [folders[i] for i in sorted(sample(xrange(len(folders)), referencenum))]
logging.debug(referencefolders)
referencedrivers = []
    
for referencefolder in referencefolders:
    referencedrivers.append(Driver(referencefolder))
    
start_reg.generatedata(referencedrivers)
start_reg.REFERENCE_DATA['prior_scores']=start_reg.parse_prior_scores('/Users/zhelezov/coding/python/telematics/axa-telematics/out/pyRegression_23_14_March_12_2015.csv')

seed(42)
pred = np.array([])
start_reg.REFERENCE_DATA["num_ref_drivers"] = 4
#for perc in [p/100.0 for p in range(60,100,10)]:
#    start_reg.REFERENCE_DATA['top_scores_percent'] = perc
#    start_reg.grid_search(os.environ['TELEMATICS'] + "1048")


start_reg.REFERENCE_DATA['top_scores_percent'] = 0.84
for i in range(0, 5):
    cv_, cv_, p = start_reg.grid_search(os.environ['TELEMATICS'] + "1048")
    logging.info("CV score for %d is %.3f", i, cv_)
# cv_1, cv_2, pr = start_reg.grid_search("/Users/zhelezov/coding/octave/telematics/Telematics/drivers/1048")
#    pred = np.hstack((pred, pr))
#    logging.info("Round 1: %.3f, Round 2: %.3f", cv_1, cv_2)



#pred = np.reshape(pred, (200, -1))
#logger.info("%s", pred)

#logger.info("%s", np.array([np.mean(pred, axis = 1), np.std(pred, axis=1)]))



