import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import copy

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyloon.multiinputloon import MultiInputLoon as Loon
from pyutils.pyutils import parsekw, hash3d, hash4d, rng, downsize

class LoonPathPlanner:
    def __init__(self, *args, **kwargs):
        field = parsekw(kwargs, 'field', 0.0) # wind field object
        res = parsekw(kwargs, 'res', 100)     # grid resolution in m
        lower = parsekw(kwargs, 'lo', 0.0)
        upper = parsekw(kwargs, 'hi', 100)
        self.sounding = parsekw(kwargs, 'sounding', False)
        self.nodes = dict()
        self.edges = dict()
        self.backedges = dict()
        self.values = dict()
        self.leaves = np.array([[np.inf, np.inf, np.inf, np.inf]])
        self.branch_length = 180 # seconds
        self.retrain(field=field, res=res, lower=lower, upper=upper)

    def retrain(self, *args, **kwargs):
        field = parsekw(kwargs, 'field', 0.0) # wind field object
        res = parsekw(kwargs, 'res', 100)     # grid resolution in m
        lower = parsekw(kwargs, 'lower', 0.0)
        upper = parsekw(kwargs, 'upper', 100)

        fx = np.zeros(len(field.field.keys()))
        fy = np.zeros(len(field.field.keys()))
        coords = np.zeros([len(field.coords.keys()),len(field.coords.values()[0])])
        for i in range(len(field.field.keys())):
            this_key = field.field.keys()[i]
            magnitude, direction = field.field[this_key]
            fx[i] = np.array([magnitude * np.cos(direction)])
            fy[i] = np.array([magnitude * np.sin(direction)])
            coords[i] = np.array(field.coords[this_key])
        self.mask = downsize(coords)
        coords = coords[:,self.mask]
        kernel = C(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-3, 1e0)) \
                    + C(1.0, (1e-3, 1e3)) * RBF(15, (1e1, 1e3)) \
                    + WhiteKernel(1, (1,1))
        self.GPx = GPR(kernel=kernel, n_restarts_optimizer=9)
        self.GPy = GPR(kernel=kernel, n_restarts_optimizer=9)
        print("Training GPx")
        self.GPx.fit(np.atleast_2d(coords), np.atleast_2d(fx).T)
        print("Training GPy")
        self.GPy.fit(np.atleast_2d(coords), np.atleast_2d(fy).T)

        return True

    def predict(self, p):
        mean_fx, std_fx = self.GPx.predict(np.atleast_2d(np.array(p)[self.mask]), return_std=True)
        mean_fy, std_fy = self.GPy.predict(np.atleast_2d(np.array(p)[self.mask]), return_std=True)
        return mean_fx, mean_fy, std_fx, std_fy

    def ev(self, p):
        mean_fx = self.GPx.predict(np.atleast_2d(np.array(p)[self.mask]), return_std=False)
        mean_fy = self.GPy.predict(np.atleast_2d(np.array(p)[self.mask]), return_std=False)
        return mean_fx, mean_fy

    def __add_node__(self, p, loon):
        self.nodes[hash3d(p)] = copy.deepcopy(loon)
        return True

    def __add_cost__(self, p, pstar):
        prev_p = 0
        prev_w = 0
        curr_w = 0
        if hash3d(p) in self.backedges:
            prev_p = self.backedges[hash3d(p)][0]
            curr_w = self.backedges[hash3d(p)][2]
        else:
            curr_w = self.__cost__(p, pstar)
        if prev_p in self.values:
            prev_w = self.values[prev_p]
        self.values[hash3d(p)] = curr_w + prev_w
        return True

    def __add_leaf__(self, p):
        val = self.values[hash3d(p)]
        self.leaves = np.append(self.leaves, [[p[0], p[1], p[2], val]], axis=0)
        return True

    def __cost__(self, p, pstar):
        return np.linalg.norm(np.subtract(p[0:2],pstar[0:2]))

    def __climb_branch__(self, loon, u, pstar, depth):
        working_loon = copy.deepcopy(loon)
        w = 0
        c_noise = 1e-5
        for i in range(int(np.ceil(self.branch_length * working_loon.Fs))):
            # Get expected value of wind at each propogating position
            expected_value = np.squeeze(self.ev(working_loon.get_pos()))
            vwind_x = expected_value[0] * np.cos(expected_value[1])
            vwind_y = expected_value[0] * np.sin(expected_value[1])
            # vl = np.subtract(vwind, working_loon.get_vel())
            # Update positions based on predicted drag force with random disturbances
            lp = working_loon.update(   vz=u + rng(1e-3),
                                        vx=vwind_x,
                                        vy=vwind_y)
                                        # fx=self.__drag_force__(working_loon, vl[0][0]) + rng(c_noise),
                                        # fy=self.__drag_force__(working_loon, vl[1][0]) + rng(c_noise),
                                        # fz=self.__drag_force__(working_loon, 0) + rng(c_noise))

            # Calculate weights of these edges
            w += self.__cost__(lp, pstar)

        x, y, z = loon.get_pos()
        wlx, wly, wlz = working_loon.get_pos()
        self.edges[hash4d([x,y,z,u])] = [hash3d([wlx, wly, wlz]), w]
        self.backedges[hash3d([wlx,wly,wlz])] = [hash3d([x,y,z]), u, w]

        self.__montecarlo__(working_loon, pstar, depth)

    def montecarlo(self, loon, pstar, depth):
        self.__reset__()
        self.__montecarlo__(loon, pstar, depth)

    def __montecarlo__(self, loon, pstar, depth):
        # Add this node and get its value
        x, y, z = loon.get_pos()
        self.__add_node__([x,y,z], loon)
        self.__add_cost__([x,y,z], pstar)

        # Base case
        if depth <= 0:
            self.__add_leaf__([x,y,z])
            return

        u = 5
        self.__climb_branch__(loon, u, pstar, depth-1)
        self.__climb_branch__(loon, 0, pstar, depth-1)
        self.__climb_branch__(loon, -u, pstar, depth-1)

    def policy(self):
        minval = np.inf
        for leaf in self.leaves:
            if leaf[3] < minval:
                leafiest = leaf
                minval = leaf[3]
        p = hash3d(leafiest[0:3])
        pol = np.array([np.inf])
        while p in self.backedges:
            pol = np.append(pol,self.backedges[p][1])
            p = self.backedges[p][0]
        return pol

    def __reset__(self):
        self.nodes = dict()
        self.edges = dict()
        self.backedges = dict()
        self.values = dict()
        self.leaves = np.array([[np.inf, np.inf, np.inf, np.inf]])
        return True

    # TODO: change to be wind-relative velocity
    def __drag_force__(self, loon, v):
        rho = 0.25 # air density
        return v * abs(v) * rho * loon.A * loon.Cd / 2
