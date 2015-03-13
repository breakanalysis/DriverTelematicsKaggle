import numpy as np
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from random import sample
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import LinearSVC
from scipy import interp

from sklearn.decomposition import RandomizedPCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer

import logging
import time

from sklearn.grid_search import GridSearchCV

from sklearn.pipeline import Pipeline


def calibrate_scores(probas):
    even_spaced = np.linspace(0,1,200)
    sorted_orig_indices = sorted(range(200), key =lambda(x): probas[x])
    sorted_indices = sorted(range(200), key = lambda(x): sorted_orig_indices[x])
    return even_spaced[sorted_indices]


def grid_search_all(X, y):
    logging.debug("Performing grid search ...")
    pipe = Pipeline(steps=[
        ('imputer', Imputer()),
        ('preprocessing', StandardScaler(copy=True, with_mean=True, with_std=True)),
        ('pca', RandomizedPCA(n_components=100)),
        ('classification', RandomForestClassifier(n_estimators=100, max_depth=4))
    ])

    cv = StratifiedKFold(y, n_folds=5, shuffle=True)

    clf = GridSearchCV(pipe,
        dict(pca__n_components = [200],
            classification__n_estimators = [200],
            classification__max_features = ["auto"],
            classification__max_depth = [20]),
            cv = cv,
        scoring='roc_auc', verbose=0, n_jobs=1)
    
    logging.debug("fit...")
    clf.fit(X, y)
    logging.debug("fit ok")
    
    logging.debug("Best parameters set found on development set:")
    
    logging.debug(clf.best_estimator_)
    
    logging.debug("Grid scores on development set:")
    
    for params, mean_score, scores in clf.grid_scores_:
        logging.debug("%0.3f (+/-%0.03f) for %r",
              mean_score, scores.std() / 2, params)
    
    logging.debug("Best score %0.3f: ", clf.best_score_)

    return clf


class RegressionDriver(object):
    """Class for Regression-based analysis of Driver traces"""

    def __init__(self, driver, datadict):
        """Initialize by providing a (positive) driver example and a dictionary of (negative) driver references."""
        self.driver = driver
        self.dict = datadict
        self.numfeatures = self.driver.num_features
        featurelist = []
        self.__indexlist = []
        for trace in self.driver.traces:
            self.__indexlist.append(trace.identifier)
            featurelist.append(trace.features)

        self.my_traindata = np.asarray(featurelist)
        self.my_testdata = np.asarray(featurelist)
        self.my_trainlabels = np.ones((self.my_traindata.shape[0],))
        self.__datadict = datadict

        self.generate_training_data()

        self.classifier = Pipeline(steps=[
            ('imputer', Imputer()),
            ('preprocessing', StandardScaler(copy=True, with_mean=True, with_std=True)),
            ('pca', RandomizedPCA(n_components=200, whiten=True)),
            ('classification', RandomForestClassifier(n_estimators=200, max_depth=20, 
                max_features="auto"))
        ])


    def generate_training_data(self):
        datadict = self.__datadict
        driver = self.driver

        data = np.empty((0, driver.num_features), float)

        setkeys = datadict["drivers"].keys()
        if driver.identifier in setkeys:
            setkeys.remove(driver.identifier)
        #else:
        setkeys = sample(setkeys, datadict["num_ref_drivers"])

        logging.info("Reference driver keys: %s", setkeys)

        for key in setkeys:
            if key != driver.identifier:
                data = np.append(data, np.asarray(datadict["drivers"][key]), axis=0)

        self.__traindata = np.append(self.my_traindata, data, axis=0)
        self.__trainlabels = np.append(self.my_trainlabels, np.zeros((data.shape[0],)), axis=0)
        
        self.__y = np.ones((self.my_testdata.shape[0],))
        self.__testdata = self.my_traindata
            
    def grid_search(self):
        """
        Grid search for the two-round strategy, see the comment in classify()
        """
        self.generate_training_data()
        clf = grid_search_all(self.__traindata, self.__trainlabels)

        cv_score_round_1 = clf.best_score_

        logging.debug(cv_score_round_1)

        estimator = clf.best_estimator_
        predict = estimator.predict_proba(self.__testdata)[:,1]
        samples = self.__trainlabels.shape[0]
        logging.debug("First round predictions %s", predict)
        ind = predict.argsort()[-samples/4:]
        logging.debug("Top 25 indices: %s", ind)
        #add more indices
        samples = self.__trainlabels.shape[0]
        zero_ind = np.arange(predict.shape[0], predict.shape[0] + samples/2)
        ind = np.append(ind, zero_ind)
        logging.debug("Second round, indices: %s", ind)
        X = self.__traindata[ind]
        y = self.__trainlabels[ind]

        logging.debug("X, y: %s %s", X, y)

        clf = grid_search_all(X, y)

        cv_score_round_2 = clf.best_score_

        logging.debug(cv_score_round_2)

        predict = clf.best_estimator_.predict_proba(self.__testdata)[:,1]
        logging.debug("Second round predictions %s", predict)
        logging.debug("done")

        return cv_score_round_1, cv_score_round_2, predict
        

    def classify(self):
        no_attempts = self.__datadict["no_classifiers"]
        attempts = np.empty((self.__y.shape[0], 0), float)

        for i in range(0, no_attempts):
            logging.info("Classification %d for driver %s", i, self.driver.identifier)
            self.generate_training_data()
            probs = self.classify_once()
            probs = np.reshape(probs, (self.__y.shape[0], -1))
            attempts = np.append(attempts, probs, axis=1)
            logging.debug("Results so far:\n %s", attempts)

        self.__y = np.median(attempts, axis=1)

        logging.debug("Final results:\n %s", self.__y)

        return self.__y


    def classify_once(self):
        """Perform classification"""
        logging.info("Performing classification for driver %d", self.driver.identifier)
        tic = time.time()
      
        clf = self.classifier
        clf.fit(self.__traindata, self.__trainlabels)
        toc1 = time.time()
        logging.info("Fitting for driver %d complete in %.2fs. Predicting first round", self.driver.identifier, (toc1 - tic))
        
        predict = clf.predict_proba(self.__testdata)[:,1]
        samples = self.__trainlabels.shape[0]
        """
        First, we predict probabilities based on the whole set of driver traces.
        On the second round we pick only 25 perc. top traces and use it as a new
        learning set hoping that the first round prediction rate is better than 0.5 
        """
        logging.debug("First round predictions %s", predict)
        ind = predict.argsort()[-samples/4:]
        #add more indices
        samples = self.__trainlabels.shape[0]
        zero_ind = np.arange(predict.shape[0], predict.shape[0] + samples/2)
        ind = np.append(ind, zero_ind)
        #logging.debug("Second round, indices: %s", ind)
        
        X = self.__traindata[ind]
        y = self.__trainlabels[ind]

        clf.fit(X, y)

        probas = clf.predict_proba(self.__testdata)[:,1]
        toc2 = time.time()
        logging.info("Predicting for driver %d complete in %.2fs", self.driver.identifier, (toc2 - tic))
        
        return probas   

    def toKaggle(self):
        """Return string in Kaggle submission format"""
        returnstring = ""
        self.__y = calibrate_scores(self.__y)
        for i in xrange(len(self.__indexlist) - 1):
            #logging.debug("%s %s %.3f\n",self.driver.identifier, self.__indexlist[i], self.__y[i])
            returnstring += "%s_%s,%.3f\n" % (self.driver.identifier, self.__indexlist[i], self.__y[i])
        returnstring += "%s_%s,%.3f" % (self.driver.identifier, self.__indexlist[len(self.__indexlist)-1], self.__y[len(self.__indexlist)-1])
        return returnstring
