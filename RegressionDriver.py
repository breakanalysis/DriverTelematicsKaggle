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
        cv=cv,
        scoring='roc_auc', verbose=50, n_jobs=1)
    
    logging.debug("fit...")
    clf.fit(X, y)
    logging.debug("fit ok")
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()
    print("Best score %0.3f: " % clf.best_score_)

    return clf


class RegressionDriver(object):
    """Class for Regression-based analysis of Driver traces"""

    def __init__(self, driver, datadict):
        """Initialize by providing a (positive) driver example and a dictionary of (negative) driver references."""
        self.driver = driver
        self.numfeatures = self.driver.num_features
        featurelist = []
        self.__indexlist = []
        for trace in self.driver.traces:
            self.__indexlist.append(trace.identifier)
            featurelist.append(trace.features)
        # Initialize train and test np arrays
        self.__traindata = np.asarray(featurelist)
        self.__testdata = np.asarray(featurelist)
        self.__trainlabels = np.ones((self.__traindata.shape[0],))
        data = np.empty((0, driver.num_features), float)
        setkeys = datadict.keys()
        if driver.identifier in setkeys:
            setkeys.remove(driver.identifier)
        else:
            setkeys = sample(setkeys, len(setkeys) - 1)
        for key in setkeys:
            if key != driver.identifier:
                data = np.append(data, np.asarray(datadict[key]), axis=0)
        self.__traindata = np.append(self.__traindata, data, axis=0)
        self.__trainlabels = np.append(self.__trainlabels, np.zeros((data.shape[0],)), axis=0)
        self.__y = np.ones((self.__testdata.shape[0],))

        self.classifier = Pipeline(steps=[
            ('imputer', Imputer()),
            ('preprocessing', StandardScaler(copy=True, with_mean=True, with_std=True)),
            ('pca', RandomizedPCA(n_components=200, whiten=True)),
            ('classification', RandomForestClassifier(n_estimators=200, max_depth=20, 
                max_features="auto"))
        ])

            
    def grid_search(self):
        """
        Grid search for the two-round strategy, see the comment in classify()
        """
        clf = grid_search_all(self.__traindata, self.__trainlabels)

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

        predict = clf.best_estimator_.predict_proba(self.__testdata)
        logging.debug("Second round predictions %s", predict)
        logging.debug("done")
        

    def cv_score(self, plot = False):
        """Perform cross-validation"""
        self.transform_data()
        y = self.__trainlabels
        X = self.__traindata

        logging.info("Started CV for %d", self.driver.identifier)

        cv = StratifiedKFold(y, n_folds=5, shuffle=True)
        classifier = self.classifier
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        
        logging.debug("Train data: %s", X)
        logging.debug("Label data: %s", y)

        for i, (train, test) in enumerate(cv):
            logging.info(" start CV iteration %d", i)

            fft = classifier.fit(X[train], y[train])
            probas_ = fft.predict(X[test])
            logging.debug("predictions: %s", probas_)
            logging.debug("labels: %s", y[test] )
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
           
            logging.debug("roc_auc for iteration %d is %.3f", i, roc_auc)
            if (plot):
                plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
            
        
        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        
        if (plot):
            plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

            plt.plot(mean_fpr, mean_tpr, 'k--',
                     label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()

        return mean_auc


    def classify(self):
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

        self.__y = clf.predict_proba(self.__testdata)[:,1]
        toc2 = time.time()
        logging.info("Predicting for driver %d complete in %.2fs", self.driver.identifier, (toc2 - tic))
        

    def toKaggle(self):
        """Return string in Kaggle submission format"""
        returnstring = ""
        for i in xrange(len(self.__indexlist) - 1):
            logging.debug("%s %s %.3f\n",self.driver.identifier, self.__indexlist[i], self.__y[i])
            returnstring += "%s_%s,%.3f\n" % (self.driver.identifier, self.__indexlist[i], self.__y[i])
        returnstring += "%s_%s,%.3f" % (self.driver.identifier, self.__indexlist[len(self.__indexlist)-1], self.__y[len(self.__indexlist)-1])
        return returnstring
