
import array
import numpy as np
import warnings
import scipy.sparse as sp

from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier, _ConstantPredictor
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.utils.validation import check_consistent_length
from sklearn.externals.joblib import Parallel, delayed

iris = datasets.load_iris()
X, y = iris.data, iris.target

# OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)

# _fit_binary and _fit_ovo_binary can not be a method,
# otherwise it's not possible to use multithread

def _fit_binary(ovo, estimator, X, y, classes=None):
    """Fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(X, y)

        # X is the training data, change to a test data
        score = (classes[0], classes[1], estimator.fit(X, y).score(X, y))
        ovo.binary_scores.append(score)

    return estimator


def _fit_ovo_binary(ovo, estimator, X, y, i, j):
    """Fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y_binary = np.empty(y.shape, np.int)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    ind = np.arange(X.shape[0])
    return _fit_binary(ovo, estimator, X[ind[cond]], y_binary, classes=[i, j])


class OneVsOneClassifierAdapted(OneVsOneClassifier):

    def __init__(self, estimator, n_jobs=1):
        """
        self.binary_scores : 
            List of tuples -> [(class, class, score),]
            It contains 'n_classes * (n_classes - 1) / 2' elements
                each one with a pair of classes and the score obtained
                by self.estimator
        """
        super(OneVsOneClassifierAdapted, self).__init__(estimator, n_jobs)
        self.binary_scores = []

    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : array-like, shape = [n_samples]
            Multi-class targets.

        Returns
        -------
        self
        """
        y = np.asarray(y)
        check_consistent_length(X, y)

        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_ovo_binary)(
                self, self.estimator, X, y, self.classes_[i], self.classes_[j])
            for i in range(n_classes) for j in range(i + 1, n_classes))

        return self
