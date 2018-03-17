import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor as KNR
import copy

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng, downsize
from pyutils import pyutils

class FieldEstimator:
    def __init__(self, *args, **kwargs):
        self.estimators = dict()
        self.X = dict()
        self.y = dict()

    def fit(self, *args, **kwargs):
        key = parsekw(kwargs, 'key', None)
        X = self.X[key]
        y = self.y[key]
        X = X.reshape(-1,1) if len(X.shape) == 1 else X
        y = y.reshape(-1,1)
        self.estimators[key].fit(X, y)

    def predict(self, *args, **kwargs):
        key = parsekw(kwargs, 'key', None)
        p = parsekw(kwargs, 'p', None)
        return_std = parsekw(kwargs, 'return_std', False)
        p = np.array(p)
        p = p.reshape(-1,1) if len(p.shape) == 1 else p
        if return_std:
            prediction, std = self.estimators[key].predict(p, return_std=return_std)
            return prediction, std
        else:
            prediction = self.estimators[key].predict(p, return_std=return_std)
            return prediction

    def changing_estimators(self, *args, **kwargs):
        return False

class GPFE(FieldEstimator):
    def __init__(self, *args, **kwargs):
        FieldEstimator.__init__(self)
        default_kernel = C(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-6, 1e0))
        kernel = parsekw(kwargs, 'kernel', default_kernel)
        n_restarts_optimizer = parsekw(kwargs, 'n_restarts_optimizer', 9)
        self.estimators[0] = GPR(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)

    def fit(self, *args, **kwargs):
        key = 0
        self.X[key] = parsekw(kwargs, 'X', None)
        self.y[key] = parsekw(kwargs, 'y', None)
        FieldEstimator.fit(self, key=key)
        if len(self.X[key][0]) > 1:
            self.__partition__(self.X[key], self.y[key])

    def predict(self, *args, **kwargs):
        key = 0
        return FieldEstimator.predict(self,
                                        key=key,
                                        p=kwargs.get('p'),
                                        return_std=kwargs.get('return_std'))

    def __partition__(self, X, y):
        self.estimator_locations = dict()
        idx = 0
        for i in range(len(X)):
            xcoord = np.int(X[i][0])
            ycoord = np.int(X[i][1])
            zcoord = 0
            p = np.array([xcoord, ycoord, zcoord])
            key = hash3d(p)
            if key in self.X.keys():
                self.X[key] = np.append(self.X[key], X[i][2])
                self.y[key] = np.append(self.y[key], y[i])
            else:
                self.X[key] = X[i][2]
                self.y[key] = y[i]
                self.estimator_locations[key] = p
                for key in self.X.keys():
                    self.X[key] = np.atleast_2d(self.X[key]).T
            self.prediction_key = key

    def changing_estimators(self, *args, **kwargs):
        if len(self.X[0][0]) == 1:
            return False
        pos = np.atleast_2d(parsekw(kwargs, 'p', None))
        radius = parsekw(kwargs, 'radius', 40000)
        keys = self.estimator_locations.keys()
        old_loc = self.estimator_locations[self.prediction_key]
        oldx = old_loc[0]
        oldy = old_loc[1]
        d_old = (oldx - pos[0,0])**2 + (oldy - pos[0,1])**2
        for i, key in enumerate(keys):
            dx = self.estimator_locations[key][0] - pos[0,0]
            dy = self.estimator_locations[key][1] - pos[0,1]
            d = dx**2 + dy**2
            if d < radius**2 and d < d_old:
                if key != self.prediction_key:
                    self.prediction_key = key
                    return True
        return False

class Multi1DGP(FieldEstimator):
    def __init__(self, *args, **kwargs):
        self.prediction_key = 0.0
        FieldEstimator.__init__(self)

    def fit(self, *args, **kwargs):
        X = parsekw(kwargs, 'X', None)
        y = parsekw(kwargs, 'y', None)
        default_kernel = C(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-6, 1e0))
        kernel = parsekw(kwargs, 'kernel', default_kernel)
        n_restarts_optimizer = parsekw(kwargs, 'n_restarts_optimizer', 9)
        print("\t\tpartitioning...")
        self.__partition__(X, y)
        i = 0
        for key in self.estimator_locations.keys():
            self.prediction_key = key
            i += 1
            print("\t\t" + str(i) + "/" + str(len(self.estimator_locations.keys())) + \
            "\t(" + str(self.estimator_locations[key][0]) + ", " + \
            str(self.estimator_locations[key][1]) + ")")
            self.estimators[key] = GPR(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
            FieldEstimator.fit(self, key=key)

    def predict(self, *arg, **kwargs):
        radius = parsekw(kwargs, 'radius', 40000)
        p = np.atleast_2d(parsekw(kwargs, 'p', None))
        self.changing_estimators(p=p, radius=radius)
        return FieldEstimator.predict(self,
                                    key=self.prediction_key,
                                    p=p[:,2],
                                    return_std=kwargs.get('return_std'))

    def __partition__(self, X, y):
        self.estimator_locations = dict()
        idx = 0
        for i in range(len(X)):
            xcoord = np.int(X[i][0])
            ycoord = np.int(X[i][1])
            zcoord = 0
            p = np.array([xcoord, ycoord, zcoord])
            key = hash3d(p)
            if key in self.X.keys():
                self.X[key] = np.append(self.X[key], X[i][2])
                self.y[key] = np.append(self.y[key], y[i])
            else:
                self.X[key] = X[i][2]
                self.y[key] = y[i]
                self.estimator_locations[key] = p
                for key in self.X.keys():
                    self.X[key] = np.atleast_2d(self.X[key]).T

    def changing_estimators(self, *args, **kwargs):
        pos = np.atleast_2d(parsekw(kwargs, 'p', None))
        radius = parsekw(kwargs, 'radius', 40000)
        keys = self.estimator_locations.keys()
        old_loc = self.estimator_locations[self.prediction_key]
        oldx = old_loc[0]
        oldy = old_loc[1]
        d_old = (oldx - pos[0,0])**2 + (oldy - pos[0,1])**2
        for i, key in enumerate(keys):
            dx = self.estimator_locations[key][0] - pos[0,0]
            dy = self.estimator_locations[key][1] - pos[0,1]
            d = dx**2 + dy**2
            if d < radius**2 and d < d_old:
                if key != self.prediction_key:
                    self.prediction_key = key
                    return True
        return False

class KNN1DGP(Multi1DGP):
    def predict(self, *arg, **kwargs):
        pos = np.atleast_2d(parsekw(kwargs, 'p', None))
        n_neighbors = parsekw(kwargs, 'n_neighbors', 4)
        keys = self.estimator_locations.keys()
        prediction = np.zeros(len(pos))
        X = np.zeros([len(keys), 2])
        y = np.zeros([len(keys), len(pos)])
        for i, key in enumerate(keys):
            pred = FieldEstimator.predict(self, key=key, p=pos[:,2], return_std=False)
            y[i] = pred.reshape(-1)
            X[i] = self.estimator_locations[key][0:2]
        for i, p in enumerate(pos):
            knn = KNR(n_neighbors=n_neighbors, weights='distance')
            knn.fit(X, y[:,i])
            prediction[i] = knn.predict(np.atleast_2d(p[0:2]))
        return prediction

    def changing_estimators(self, *args, **kwargs):
        return True
