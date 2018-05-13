import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Arrow
from matplotlib.collections import PatchCollection
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor as KNR
import copy

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng, downsize
from pyutils import pyutils
from pyutils.constants import DEGLAT_2_KM

class FieldEstimator:
    def __init__(self, *args, **kwargs):
        self.estimators = dict()
        self.recently_sampled_X = dict()
        self.recently_sampled_y = dict()
        self.expiring_X = dict()
        self.expiring_y = dict()
        self.X = dict()
        self.y = dict()

    def __reset__(self, key):
        if key in self.recently_sampled_X.keys():
            self.expiring_X[key] = copy.deepcopy(self.recently_sampled_X[key])
            self.expiring_y[key] = copy.deepcopy(self.recently_sampled_y[key])
            self.recently_sampled_X[key] = []
            self.recently_sampled_y[key] = []

    def __build_data__(self, key):
        X = self.X[key]
        y = self.y[key]
        # if self.reset:
            # self.__reset__(key)
        if key in self.recently_sampled_X.keys():
            X = np.append(X, self.recently_sampled_X[key])
            y = np.append(y, self.recently_sampled_y[key])
        if key in self.expiring_X.keys():
            X = np.append(X, self.expiring_X[key])
            y = np.append(y, self.expiring_y[key])
        X = X.reshape(-1,1) if len(X.shape) == 1 else X
        y = y.reshape(-1,1)
        return X, y

    def fit(self, *args, **kwargs):
        key = parsekw(kwargs, 'key', None)
        self.reset = parsekw(kwargs, 'reset', True)
        X, y = self.__build_data__(key)
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

    def add_data(self, *args, **kwargs):
        key = parsekw(kwargs, 'key', None)
        new_X = parsekw(kwargs, 'X', None)
        new_y = parsekw(kwargs, 'y', None)
        if key in self.recently_sampled_X.keys():
            self.recently_sampled_X[key] = np.append(self.recently_sampled_X[key], new_X)
            self.recently_sampled_y[key] = np.append(self.recently_sampled_y[key], new_y)
        else:
            self.recently_sampled_X[key] = new_X
            self.recently_sampled_y[key] = new_y

    def changing_estimators(self, *args, **kwargs):
        return False

    def __set_prediction_key__(self, key):
        # if self.prediction_key in self.estimator_locations.keys():
        #     if key in self.estimator_locations.keys():
        #         print("Changing from " + str(self.estimator_locations[self.prediction_key]) + \
        #                 " to " + str(self.estimator_locations[key]))
        self.prediction_key = key

    def get_current_estimator(self):
        return self.estimators[self.prediction_key]

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
        # Partition the data just so we can detect when we need to rebuild
        # the jetstreams (does not affect GP)
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
            FieldEstimator.__set_prediction_key__(self, key)

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
                    print("C H A N G I N G !")
                    FieldEstimator.__set_prediction_key__(self, key)
                    return True
        return False

class Multi1DGP(FieldEstimator):
    def __init__(self, *args, **kwargs):
        self.prediction_key = 0.
        FieldEstimator.__init__(self)
        self.n_closest = np.array([])
        self.fitted = np.array([])

    def fit(self, *args, **kwargs):
        X = parsekw(kwargs, 'X', None)
        y = parsekw(kwargs, 'y', None)
        p = parsekw(kwargs, 'p', np.zeros(3))
        restrict_data = parsekw(kwargs, 'restrict', False)
        restriction = parsekw(kwargs, 'restriction', 0.5)
        default_kernel = C(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-6, 1e0))
        kernel = parsekw(kwargs, 'kernel', default_kernel)
        n_restarts_optimizer = parsekw(kwargs, 'n_restarts_optimizer', 9)
        print("\t\tpartitioning...")
        self.X = dict()
        self.y = dict()
        self.n_closest = np.array([])
        self.fitted = np.array([])
        self.__partition__(X, y)
        if restrict_data:
            self.__restrict__(restriction)
        # for i, key in enumerate(self.n_closest):
        for i, key in enumerate(self.estimator_locations.keys()):
            FieldEstimator.__set_prediction_key__(self, key)
            # print progress and lateral coordinates of this estimator
            # print("\t\t" + str(i) + "/" + str(len(self.estimator_locations.keys())) + \
            #     "\t(" + str(self.estimator_locations[key][0]) + ", " + \
            #     str(self.estimator_locations[key][1]) + ")")
            FieldEstimator.__reset__(self, key)
            self.estimators[key] = GPR(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
            # FieldEstimator.fit(self, key=key, reset=True)
            # self.fitted = np.append(self.fitted, key)
        self.changing_estimators(p=p)
        # self.__fit_n_closest__(p, 1)

    def predict(self, *arg, **kwargs):
        radius = parsekw(kwargs, 'radius', 40000)
        p = np.atleast_2d(parsekw(kwargs, 'p', None))
        self.changing_estimators(p=p, radius=radius)
        # self.__fit_n_closest__(p, 4)
        return FieldEstimator.predict(self,
                                    key=self.prediction_key,
                                    p=p[:,2],
                                    return_std=kwargs.get('return_std'))

    def __restrict__(self, restriction):
        if restriction == 1:
            return
        if restriction < 1:
            if restriction > 0.5:
                print("WARNING: restriction between 0.5 and 1.0 is meaningless, taking no action")
                return
            else:
                n_to_keep = 1
                n_to_skip = np.int(np.round( 1. / restriction ))
        else:
            n_to_keep = restriction
            n_to_skip = 1

        for i, key in enumerate(self.X.keys()):
            skip_counter = 0
            keep_counter = 0
            # Sort data in order of altitude
            sort_idx = np.argsort(self.X[key])
            X_curr = self.X[key][sort_idx]
            y_curr = self.y[key][sort_idx]
            X_new = []
            y_new = []
            keeping = True
            for j, x in enumerate(X_curr):
                if keeping:
                    X_new.append(X_curr[j])
                    y_new.append(y_curr[j])
                    keep_counter += 1
                    if keep_counter == n_to_keep:
                        keeping = False
                        keep_counter = 0
                else:
                    skip_counter += 1
                    if skip_counter == n_to_skip:
                        keeping = True
                        skip_counter = 0
            self.X[key] = np.array(X_new)
            self.y[key] = np.array(y_new)

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

    def __fit_n_closest__(self, p, n):
        p = np.atleast_2d(np.squeeze(p))
        distances = np.inf * np.ones(n)
        keys = -np.ones(n)
        max_idx = np.where(distances == np.max(distances))[0][0]
        for i, key in enumerate(self.estimator_locations.keys()):
            dx = self.estimator_locations[key][0] - p[0,0]
            dy = self.estimator_locations[key][1] - p[0,1]
            d = dx**2 + dy**2
            if d < distances[max_idx]:
                distances[max_idx] = d
                keys[max_idx] = key
                max_idx = np.where(distances == np.max(distances))[0][0]
        self.n_closest = keys
        for i, key in enumerate(self.n_closest):
            if key not in self.fitted:
                print("\t\t(" + str(self.estimator_locations[key][0]) + ", " + \
                    str(self.estimator_locations[key][1]) + ")")
                FieldEstimator.fit(self, key=key, reset=True)
                self.fitted = np.append(self.fitted, key)

    def changing_estimators(self, *args, **kwargs):
        pos = np.atleast_2d(np.squeeze(parsekw(kwargs, 'p', None)))
        radius = parsekw(kwargs, 'radius', 40000)
        self.__fit_n_closest__(pos, 1)
        keys = self.estimator_locations.keys()
        old_loc = self.estimator_locations[self.prediction_key]
        oldx = old_loc[0]
        oldy = old_loc[1]
        d_old = (oldx - pos[0,0])**2 + (oldy - pos[0,1])**2
        return_val = False
        for i, key in enumerate(keys):
            dx = self.estimator_locations[key][0] - pos[0,0]
            dy = self.estimator_locations[key][1] - pos[0,1]
            d = dx**2 + dy**2
            if d < radius**2 and d < d_old:
                if key != self.prediction_key:
                    FieldEstimator.__set_prediction_key__(self, key)
                    d_old = d
                    return_val = True
        return return_val

    def add_data(self, *args, **kwargs):
        FieldEstimator.add_data(self,
                                key=self.prediction_key,
                                X=kwargs.get('X'),
                                y=kwargs.get('y'))
        FieldEstimator.fit(self, key=self.prediction_key, reset=False)

    def plot(self, ax):
        p0_notfitted = []
        p1_notfitted = []
        patches_notfitted = []
        p0_fitted = []
        p1_fitted = []
        patches_fitted = []
        width = 0.5 * DEGLAT_2_KM
        height = 0.5 * DEGLAT_2_KM
        for key in self.estimator_locations.keys():
            p = self.estimator_locations[key] * 1e-3
            left = p[1] - 0.25 * DEGLAT_2_KM
            bottom = p[0] - 0.25 * DEGLAT_2_KM
            if key not in self.fitted:
                p0_notfitted.append(p[0])
                p1_notfitted.append(p[1])
                color = 'k'
                rect = Rectangle(np.array([left,bottom]), width, height, edgecolor='k', facecolor=color, alpha=0.3)
                patches_notfitted.append(rect)
            elif key != self.prediction_key:
                p0_fitted.append(p[0])
                p1_fitted.append(p[1])
                color = 'b'
                rect = Rectangle(np.array([left,bottom]), width, height, edgecolor='k', facecolor=color, alpha=0.15)
                patches_fitted.append(rect)
        collection_fitted = PatchCollection(patches_fitted, alpha=0.15, edgecolor='k', facecolor='k')
        collection_notfitted = PatchCollection(patches_notfitted, alpha=0.3, edgecolor='k', facecolor='k')
        r1 = ax.add_collection(collection_notfitted)
        r2 = ax.add_collection(collection_fitted)
        s1 = ax.scatter(p1_notfitted, p0_notfitted, c=0.5*np.ones(3), alpha=0.)
        s2 = ax.scatter(p1_fitted, p0_fitted, c='b', alpha=0.)
        loc = self.estimator_locations[self.prediction_key] * 1e-3
        left = loc[1] - 0.25 * DEGLAT_2_KM
        bottom = loc[0] - 0.25 * DEGLAT_2_KM
        patches = []
        rect = Rectangle(np.array([left,bottom]), width, height, edgecolor='k', fill=False, alpha=0.)
        patches.append(rect)
        collection = PatchCollection(patches, alpha=0., edgecolor='k')
        r3 = ax.add_collection(collection)
        s3 = ax.scatter(loc[1], loc[0], c='y', alpha=0.)
        return s1, s2, s3, r1, r2, r3

    def plot_current(self, ax, **kwargs):
        s = parsekw(kwargs, 's', 50)
        c = parsekw(kwargs, 'c', 'r')
        loc = self.estimator_locations[self.prediction_key]
        return ax.scatter(loc[1]*1e-3, loc[0]*1e-3, edgecolors=c, s=s, marker='o', facecolors='none')

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
