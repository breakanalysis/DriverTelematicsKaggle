# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets, linear_model
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, KFold

from simple_hist_features import get_training_data, get_even_training_data

###############################################################################
# Data IO and generation

all_d = np.arange(1, 2735)

d_1 = np.random.choice(all_d, 1)
all_d = np.delete(all_d, d_1)
d_0 = np.random.choice(all_d, 10, replace=False)


r_indx = np.random.choice(np.arange(1, 201), 200, replace=False)

X, y = get_training_data(d_1, d_0, r_indx)


# import some data to play with
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
#X, y = X[y != 2], y[y != 2]
#n_samples, n_features = X.shape

#print X, X.shape, y

# Add noisy features
random_state = np.random.RandomState(0)
#X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

###############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(y, n_folds=10, shuffle=True)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

#classifier = linear_model.LogisticRegression(penalty='l2', random_state=random_state)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    fft = classifier.fit(X[train], y[train])
    probas_ = fft.predict_proba(X[test])
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    # print i, roc_auc
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    %timeit

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


    
    
    
def cv_driver(d_id, n_folds=5):
    X, y = get_even_training_data(d_id)
    
    cv = StratifiedKFold(y, n_folds, shuffle=True)
    classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        fft = classifier.fit(X[train], y[train])
        probas_ = fft.predict_proba(X[test])
    
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        %timeit
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    

# <codecell>

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets, linear_model
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

from simple_hist_features import get_even_training_data, get_all_driver_features

random_state = np.random.RandomState(0)
######
def train_driver(d_id, n_iter = 10):
    X_all = get_all_driver_features(d_id)
    
    print "Computed features for all routes"
    
    probas = np.zeros((200, n_iter))
    for i in range(0, n_iter):
        X, y = get_even_training_data(d_id, 100, weight = 1)
        classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)
        fft = classifier.fit(X, y)
        predict = fft.predict_proba(X_all)
        probas[:,i] = predict[:, 1]
        ##print predict
    
    return np.mean(probas, axis=1), probas

# <codecell>

result, probas = train_driver(1)

