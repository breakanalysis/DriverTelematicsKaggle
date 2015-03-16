"""Main module for Kaggle AXA Competition

Uses the logistic regression idea described by Stephane Soulier: https://www.kaggle.com/c/axa-driver-telematics-analysis/forums/t/11299/score-0-66-with-logistic-regression
Hence, we use the traces from every driver as positive examples and build a set of references that we use as negative examples. Note that our set is larger by one driver, in case the reference set includes the driver that we are currently using as positive.
"""

from datetime import datetime
from Driver import Driver
from RegressionDriver import RegressionDriver
import os
import sys
from random import sample, seed
import logging
import re
import numpy as np

from joblib import Parallel, delayed

import time

REFERENCE_DATA = {"drivers": {}, 
    "num_ref_drivers": 4, 
    "no_classifiers": 1,
    "top_scores_percent": 0.84,
    "prior_scores": {}}


def get_driver_folder(driver_id):
    return os.path.join(os.environ['TELEMATICS'], driver_id)



def parse_prior_scores(path):
    
    #path = '/Users/jacobsznajdman/Documents/Kaggle/Telematics/workspace/pyRegression_02_58_March_07_2015.csv'

    result_hash = {}

    for line in open(path).readlines():
        m=re.match(r'(\d+)_(\d+),(.+)',line)
        if not m:
            continue
        driver_id = int(m.group(1))
        if driver_id not in result_hash:
            result_hash[driver_id] = np.empty(200)
        result_hash[driver_id][int(m.group(2))-1]=float(m.group(3))
    
    newlines = ['driver_trip,prob\n']    

    return result_hash

def generatedata(drivers):
    """
    Generates reference data for regression

    Input: List of driver folders that are read.
    Returns: Nothing, since this data is stored in global variable ReferenceData
    """
    global REFERENCE_DATA
    for driver in drivers:
        REFERENCE_DATA["drivers"][driver.identifier] = driver.generate_data_model
    
    
     


def perform_analysis(folder, cv=False):
    logging.info("Working on {0}".format(folder))

    driver_id = folder.split("/")[-1]
    logging.info("Driver id: %s", driver_id)

    if (driver_id[0] =='2'):
        logging.info("Skipping what is starting with 2")
        return ""

    sys.stdout.flush()
    temp = Driver(folder)
    cls = RegressionDriver(temp, REFERENCE_DATA)
    if (cv):
        logging.info("CV score for driver %d: %.3f", temp.identifier, cls.cv_score())
    cls.classify()
    return cls.toKaggle()

def grid_search(folder):
    logging.info("Grid search on {0}".format(folder))
    temp = Driver(folder)
    cls = RegressionDriver(temp, REFERENCE_DATA)
    return cls.grid_search()

def classify(folder, toKaggle=False):
    logging.info("Grid search on {0}".format(folder))
    temp = Driver(folder)
    cls = RegressionDriver(temp, REFERENCE_DATA)
    ret = cls.classify()
    if (toKaggle):
        logging.info(cls.toKaggle())
    return ret


def analysis(foldername, outdir, prior_path, max_drivers = 5):
    """
    Start the analysis

    Input:
        1) Path to the driver directory
        2) Path where the submission file should be written
        3) Number of drivers to compare against
    """
    seed(42)
    start = datetime.now()
    submission_id = datetime.now().strftime("%H_%M_%B_%d_%Y")
    folders = [os.path.join(foldername, f) for f in os.listdir(foldername) if os.path.isdir(os.path.join(foldername, f))]
    referencenum = 30
    referencefolders = [folders[i] for i in sorted(sample(xrange(len(folders)), referencenum))]
    logging.debug(referencefolders)
    referencedrivers = []
    
    driver_count = 0
    for referencefolder in referencefolders:
        referencedrivers.append(Driver(referencefolder))
        driver_count += 1
        if (driver_count == max_drivers) and (max_drivers > 0):
            break

    logging.info("Generating ref data")
    generatedata(referencedrivers)

    logging.info("Parsing prior scores")
    REFERENCE_DATA['prior_scores']=parse_prior_scores(prior_path)
    logging.info("Prior scores done")
    
    results = Parallel(n_jobs=4, verbose=50)(delayed(perform_analysis)(folder) for folder in folders)
    
    with open(os.path.join(outdir, "pyRegression_{0}.csv".format(submission_id)), 'w') as writefile:
        writefile.write("driver_trip,prob\n")
        for item in results:
            writefile.write("%s\n" % item)
    print 'Done, elapsed time: %s' % str(datetime.now() - start)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    logging.debug("my path %s", MyPath)
    #analysis(os.environ['TELEMATICS'], "/Users/zhelezov/coding/python/telematics/HelloWorld/axa-telematics/data/out/", 5, -1)
    
    where_big_data_lives = os.environ['TELEMATICS']
    where_the_final_file_is_saved = "/Users/zhelezov/coding/python/telematics/axa-telematics/out/"
    where_the_csv_file_lies = "/Users/zhelezov/coding/python/telematics/axa-telematics/out/pyRegression_23_14_March_12_2015.csv"

    analysis(where_big_data_lives, where_the_final_file_is_saved, where_the_csv_file_lies, -1) 
    



