import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
import copy

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyloon.multiinputloon import MultiInputLoon as Loon

class LoonPathPlanner:
    def __init__(self, *args, **kwargs):
        field = kwargs.get('field') # wind field object
        res = kwargs.get('res')     # grid resolution in m
        self.nodes = dict()
        self.edges = dict()
        self.backedges = dict()
        self.values = dict()
        self.leaves = -np.ones([1,4])
        self.retrain(field=field, res=res)

    def retrain(self, *args, **kwargs):
        field = kwargs.get('field') # wind field object
        res = kwargs.get('res')     # grid resolution in m

        # Get training points
        x = np.linspace(0, field.xdim-1, int(np.floor(field.xdim / res)))
        y = np.linspace(0, field.ydim-1, int(np.floor(field.ydim / res)))
        z = np.linspace(0, field.zdim-1, int(np.floor(field.zdim / res)))
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
        mean_fx, std_fx = self.GPx.predict(np.atleast_2d(p), return_std=True)
        mean_fy, std_fy = self.GPy.predict(np.atleast_2d(p), return_std=True)
        mean_fz, std_fz = self.GPz.predict(np.atleast_2d(p), return_std=True)
        return mean_fx, mean_fy, mean_fz, std_fx, std_fy, std_fz

    def ev(self, p):
        mean_fx = self.GPx.predict(np.atleast_2d(p), return_std=False)
        mean_fy = self.GPy.predict(np.atleast_2d(p), return_std=False)
        mean_fz = self.GPz.predict(np.atleast_2d(p), return_std=False)
        return mean_fx, mean_fy, mean_fz

    def montecarlo(self, loon, pstar, depth):

        working_loon = copy.deepcopy(loon)
        branch_length = 1 # seconds

        # Add this node and get its value
        x, y, z = working_loon.get_pos()
        self.nodes[self.hash3d([x,y,z])] = copy.deepcopy(working_loon)
        prev_p = 0
        prev_w = 0
        curr_w = 0
        if self.hash3d([x,y,z]) in self.backedges:
            prev_p = self.backedges[self.hash3d([x,y,z])][0]
            curr_w = self.backedges[self.hash3d([x,y,z])][2]
        else:
            curr_w = np.linalg.norm(np.subtract(pstar,[x,y,z]))
        if prev_p in self.values:
            prev_w = self.values[prev_p]
        self.values[self.hash3d([x,y,z])] = curr_w + prev_w

        # Base case
        if depth <= 0:
            val = self.values[self.hash3d([x,y,z])]
            self.leaves = np.append(self.leaves, [[x, y, z, val]], axis=0)
            return

        # Get loons to propogate into next nodes
        loon_neg = copy.deepcopy(working_loon)
        loon_0 = copy.deepcopy(working_loon)
        loon_pos = copy.deepcopy(working_loon)

        w_neg = 0
        w_0 = 0
        w_pos = 0
        c_noise = 0.0
        for i in range(int(np.ceil(branch_length * working_loon.Fs))):
            # Get expected value of wind at each propogating position
            lnx, lny, lnz = self.ev(loon_neg.get_pos())
            l0x, l0y, l0z = self.ev(loon_0.get_pos())
            lpx, lpy, lpz = self.ev(loon_pos.get_pos())
            #print(lnx)

            # Update positions based on predicted drag force with random disturbances
            lnp = loon_neg.update(  vz=-1,
                                    fx=self.__drag_force__(loon_neg, lnx) + self.__rand__(c_noise),
                                    fy=self.__drag_force__(loon_neg, lny) + self.__rand__(c_noise),
                                    fz=self.__drag_force__(loon_neg, lnz) + self.__rand__(c_noise))

            l0p = loon_0.update(    vz=0,
                                    fx=self.__drag_force__(loon_0, l0x) + self.__rand__(c_noise),
                                    fy=self.__drag_force__(loon_0, l0y) + self.__rand__(c_noise),
                                    fz=self.__drag_force__(loon_0, l0z) + self.__rand__(c_noise))

            lpp = loon_pos.update(  vz=1,
                                    fx=self.__drag_force__(loon_pos, lpx) + self.__rand__(c_noise),
                                    fy=self.__drag_force__(loon_pos, lpy) + self.__rand__(c_noise),
                                    fz=self.__drag_force__(loon_pos, lpz) + self.__rand__(c_noise))

            # Calculate weights of these edges
            w_neg += np.linalg.norm(np.subtract(lnp,pstar))
            w_0 += np.linalg.norm(np.subtract(l0p,pstar))
            w_pos += np.linalg.norm(np.subtract(lpp,pstar))

        # Get positions of next nodes
        lnx, lny, lnz = loon_neg.get_pos()
        l0x, l0y, l0z = loon_0.get_pos()
        lpx, lpy, lpz = loon_pos.get_pos()

        # Add forward edges to the next nodes
        self.edges[self.hash4d([x,y,z,-1])] = [self.hash3d([lnx, lny, lnz]), w_neg]
        self.edges[self.hash4d([x,y,z,0])] = [self.hash3d([l0x, l0y, l0z]), w_0]
        self.edges[self.hash4d([x,y,z,1])] = [self.hash3d([lpx, lpy, lpz]), w_pos]

        # Add backedges to this node
        self.backedges[self.hash3d([lnx,lny,lnz])] = [self.hash3d([x, y, z]), -1, w_neg]
        self.backedges[self.hash3d([l0x,l0y,l0z])] = [self.hash3d([x, y, z]), 0, w_0]
        self.backedges[self.hash3d([lpx,lpy,lpz])] = [self.hash3d([x, y, z]), 1, w_pos]

        # Recursive calls
        self.montecarlo(loon_neg, pstar, depth-1)
        self.montecarlo(loon_0, pstar, depth-1)
        self.montecarlo(loon_pos, pstar, depth-1)

    def hash3d(self, p):
        P0 = 73856093
        P1 = 19349663
        P2 = 83492791
        return (int(p[0]*P0) ^ int(p[1]*P1) ^ int(p[2]*P2))

    def hash4d(self, p):
        P0 = 73856093
        P1 = 19349663
        P2 = 83492791
        P3 = 32452843
        return (int(p[0]*P0) ^ int(p[1]*P1) ^ int(p[2]*P2) ^ int(p[3]*P3))

    # TODO: change to be wind-relative velocity
    def __drag_force__(self, loon, v):
        rho = 1 # air density
        return v * v * rho * loon.A * loon.Cd / 2

    def __rand__(self, c):
        return (0.5 - np.random.rand()) * c
