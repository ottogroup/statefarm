from nolearn.lasagne import BatchIterator
from nolearn.lasagne.base import _sldict
import numpy as np
from scipy.ndimage import rotate


class CVTrainSplit(object):
    def __init__(self, cv):
        self.cv = cv

    def __call__(self, X, y, net=None):
        train_indices, valid_indices = next(iter(self.cv))
        X_train, y_train = _sldict(X, train_indices), y[train_indices]
        X_valid, y_valid = _sldict(X, valid_indices), y[valid_indices]
        return X_train, X_valid, y_train, y_valid


class RotateBatchIterator(BatchIterator):
    def __init__(self, *args, max_angle=20, **kwargs):
        self.max_angle = max_angle
        super(RotateBatchIterator, self).__init__(*args, **kwargs)

    def transform(self, X, y):
        angle = (np.random.rand() - 0.5) * 2 * self.max_angle
        X_new = np.zeros_like(X)
        for i, x in enumerate(X):
            X_new[i, 0] = rotate(x[0], angle, mode='nearest', reshape=False)
        return X_new, y
