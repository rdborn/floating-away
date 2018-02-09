import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import copy

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyloon.multiinputloon import MultiInputLoon as Loon
from pyutils.pyutils import parsekw, hash3d, hash4d, rng

class LoonPathPlanner:
    def __init__(self, *args, **kwargs):
        field = parsekw(kwargs.get('field'), 0.0) # wind field object
        res = parsekw(kwargs.get('res'), 100)     # grid resolution in m
        lower = parsekw(kwargs.get('lo'), 0.0)
        upper = parsekw(kwargs.get('hi'), 100)
        self.sounding = parsekw(kwargs.get('sounding'), False)
        self.nodes = dict()
        self.edges = dict()
        self.backedges = dict()
        self.values = dict()
        self.leaves = np.array([[np.inf, np.inf, np.inf, np.inf]])
        self.branch_length = 180 # seconds
        self.retrain(field=field, res=res, lower=lower, upper=upper)

    def retrain(self, *args, **kwargs):
        field = parsekw(kwargs.get('field'), 0.0) # wind field object
        res = parsekw(kwargs.get('res'), 100)     # grid resolution in m
        lower = parsekw(kwargs.get('lower'), 0.0)
        upper = parsekw(kwargs.get('upper'), 100)

        # Get training points
        if self.sounding:
            initialized = False
            for i in range(len(field.field.keys())):
                this_key = field.field.keys()[i]
                if this_key > lower and this_key < upper:
                    if not initialized:
                        z = np.array([this_key])
                        fx = np.array([field.field[this_key][0] * np.cos(field.field[this_key][1])])
                        fy = np.array([field.field[this_key][0] * np.sin(field.field[this_key][1])])
                        initialized = True
                    else:
                        z = np.append(z,[this_key])
                        fx = np.append(fx, [field.field[this_key][0] * np.cos(field.field[this_key][1])])
                        fy = np.append(fy, [field.field[this_key][0] * np.sin(field.field[this_key][1])])
            L = 1
            lo = 1e-3
            hi = 1e2
            kernel = C(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-3, 1e0)) \
                        + C(1.0, (1e-3, 1e3)) * RBF(15, (1e1, 1e3)) \
                        + WhiteKernel(1, (1,1))
            self.GPx = GPR(kernel=kernel, n_restarts_optimizer=9)
            self.GPy = GPR(kernel=kernel, n_restarts_optimizer=9)

            # Train our Gaussian Gaussian Process
            print("Training GPx")
            self.GPx.fit(np.atleast_2d(z).T, np.atleast_2d(fx).T)
            print("Training GPy")
            self.GPy.fit(np.atleast_2d(z).T, np.atleast_2d(fy).T)
        else:
            x = np.linspace(lower, upper, int(np.floor((upper-lower) / res)))
            y = np.linspace(lower, upper, int(np.floor((upper-lower) / res)))
            z = np.linspace(lower, upper, int(np.floor((upper-lower) / res)))
            xyz = np.vstack(map(np.ravel, np.meshgrid(x, y, z))).T

            # Get training observations
            fx = np.zeros(len(xyz))
            fy = np.zeros(len(xyz))
            fz = np.zeros(len(xyz))
            for i in range(len(xyz)):
                mag, angle = field.get_flow(xyz[i,0], xyz[i,1], xyz[i,2])
                fx[i] = mag * np.cos(angle)
                fy[i] = mag * np.sin(angle)
                fz[i] = 0

            # Set up our Gaussian Process (yay!)
            L = 1
            lo = 1e-3
            hi = 1e2
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[L, L, L], length_scale_bounds=(lo, hi), nu=1.5)
            self.GPx = GPR(kernel=kernel, n_restarts_optimizer=3)
            self.GPy = GPR(kernel=kernel, n_restarts_optimizer=3)
            self.GPz = GPR(kernel=kernel, n_restarts_optimizer=3)

            # Train our Gaussian Gaussian Process
            print("Training GPx")
            self.GPx.fit(xyz, fx.ravel())
            print("Training GPy")
            self.GPy.fit(xyz, fy.ravel())
            print("Training GPz")
            self.GPz.fit(xyz, fz.ravel())

        return True

    def predict(self, p):
        mean_fz = 0.0
        std_fz = 0.0
        if not self.sounding:
            mean_fz, std_fz = self.GPz.predict(np.atleast_2d(p), return_std=True)
        mean_fx, std_fx = self.GPx.predict(np.atleast_2d(p), return_std=True)
        mean_fy, std_fy = self.GPy.predict(np.atleast_2d(p), return_std=True)
        return mean_fx, mean_fy, mean_fz, std_fx, std_fy, std_fz

    def ev(self, p):
        mean_fz = [0.0]
        if not self.sounding:
            mean_fz = self.GPz.predict(np.atleast_2d(p), return_std=False)
        else:
            p = p[2]
        mean_fx = self.GPx.predict(np.atleast_2d(p), return_std=False)
        mean_fy = self.GPy.predict(np.atleast_2d(p), return_std=False)
        return mean_fx, mean_fy, mean_fz

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
            curr_w = np.linalg.norm(np.subtract(pstar[0:2],p[0:2]))
        if prev_p in self.values:
            prev_w = self.values[prev_p]
        self.values[hash3d(p)] = curr_w + prev_w
        return True

    def __add_leaf__(self, p):
        val = self.values[hash3d(p)]
        self.leaves = np.append(self.leaves, [[p[0], p[1], p[2], val]], axis=0)
        return True

    def __climb_branch__(self, loon, u, pstar, depth):
        working_loon = copy.deepcopy(loon)
        w = 0
        c_noise = 1e-5
        for i in range(int(np.ceil(self.branch_length * working_loon.Fs))):
            # Get expected value of wind at each propogating position
            vl = np.subtract(self.ev(working_loon.get_pos()), working_loon.get_vel())
            # Update positions based on predicted drag force with random disturbances
            lp = working_loon.update(   vz=u,
                                        vx=vl[0][0],
                                        vy=vl[0][1])
                                        # fx=self.__drag_force__(working_loon, vl[0][0]) + rng(c_noise),
                                        # fy=self.__drag_force__(working_loon, vl[1][0]) + rng(c_noise),
                                        # fz=self.__drag_force__(working_loon, 0) + rng(c_noise))

            # Calculate weights of these edges
            w += np.linalg.norm(np.subtract(lp,pstar))

        x, y, z = loon.get_pos()
        wlx, wly, wlz = working_loon.get_pos()
        self.edges[hash4d([x,y,z,u])] = [hash3d([wlx, wly, wlz]), w]
        self.backedges[hash3d([wlx,wly,wlz])] = [hash3d([x,y,z]), u, w]

        self.montecarlo(working_loon, pstar, depth)

        return working_loon, w

    def montecarlo(self, loon, pstar, depth):
        # Add this node and get its value
        x, y, z = loon.get_pos()
        self.__add_node__([x,y,z], loon)
        self.__add_cost__([x,y,z], pstar)

        # Base case
        if depth <= 0:
            self.__add_leaf__([x,y,z])
            return

        u = 5
        loon_neg, w_neg = self.__climb_branch__(loon, u, pstar, depth-1)
        loon_0, w_0 = self.__climb_branch__(loon, 0, pstar, depth-1)
        loon_pos, w_pos = self.__climb_branch__(loon, -u, pstar, depth-1)

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

    def reset(self):
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
