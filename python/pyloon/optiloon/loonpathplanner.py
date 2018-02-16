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
        lower = parsekw(kwargs, 'lo', 0.0)
        upper = parsekw(kwargs, 'hi', 100)
        self.retrain(field=field, lower=lower, upper=upper)

    def retrain(self, *args, **kwargs):
        field = parsekw(kwargs, 'field', 0.0) # wind field object
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

    def __cost__(self, p, pstar):
        return np.linalg.norm(np.subtract(p[0:2],pstar[0:2]))

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

class MonteCarloPlanner(LoonPathPlanner):
    def __init__(self, *args, **kwargs):
        self.nodes = dict()
        self.edges = dict()
        self.backedges = dict()
        self.values = dict()
        self.leaves = np.array([[np.inf, np.inf, np.inf, np.inf]])
        self.branch_length = 180 # seconds
        LoonPathPlanner.__init__(   self,
                                    field=kwargs.get('field'),
                                    lo=kwargs.get('lo'),
                                    hi=kwargs.get('hi'))

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

    def plan(self, loon, pstar, depth):
        self.__reset__()
        self.__montecarlo__(loon, pstar, depth)
        return self.__policy__()

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

    def __policy__(self):
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

class PlantInvertingController(LoonPathPlanner):
    def plan(self, *args, **kwargs):
        loon = parsekw(kwargs, 'loon', None)
        p = np.array(loon.get_pos())
        if loon == None:
            warning("Must specify loon.")
        J_x = self.__differentiate_gp__(self.GPx, p[2])
        J_y = self.__differentiate_gp__(self.GPy, p[2])
        # f_x, f_y, trash, trash = LoonPathPlanner.predict(self, p=p)
        J = np.array([J_x, J_y])
        x = p[0:2]
        num = np.inner(J, -x / np.inner(x, x))
        den = np.inner(J,J)
        u = 10000.0 * num / den
        print("u: " + str(u))
        u = u if not np.isnan(u) else 0
        u = u if u < 5.0 else 5.0
        u = u if u > -5.0 else -5.0
        return u

    def __kstar_gp__(self, GP, xstar):
        x = GP.X_train_
        theta = np.exp(GP.kernel_.theta)
        C1 = theta[0]
        L1 = theta[1]
        C2 = theta[2]
        L2 = theta[3]
        Kstar1 =  C1 * np.exp(-((x - xstar)**2) / (2 * L1**2))
        Kstar2 =  C2 * np.exp(-((x - xstar)**2) / (2 * L2**2))
        return Kstar1, Kstar2

    def __recreate_gp__(self, GP):
        theta = np.exp(GP.kernel_.theta)
        C1 = theta[0]
        L1 = theta[1]
        C2 = theta[2]
        L2 = theta[3]
        sigma = theta[4]
        x = GP.X_train_
        y = GP.y_train_
        K1 = np.zeros([len(x),len(x)])
        K2 = np.zeros([len(x),len(x)])
        for i in range(len(x)):
            for j in range(len(x)):
                K1[i,j] = C1 * np.exp(-((x[i] - x[j])**2) / (2 * L1**2))
                K2[i,j] = C2 * np.exp(-((x[i] - x[j])**2) / (2 * L2**2))
        M = np.dot(np.linalg.inv(K1) + np.linalg.inv(K2) + np.eye(len(x))*sigma, y)
        return K1, K2

    def __differentiate_gp__(self, GP, xstar):
        K1, K2 = self.__recreate_gp__(GP)
        y = GP.y_train_
        theta = np.exp(GP.kernel_.theta)
        M1 = np.dot(np.linalg.inv(K1),y)
        M2 = np.dot(np.linalg.inv(K2),y)
        Kstar1, Kstar2 = self.__kstar_gp__(GP, xstar)
        print(np.dot(Kstar1.T,M1)+np.dot(Kstar2.T,M2))
        print(GP.predict(xstar))
        L1 = theta[1]
        L2 = theta[3]
        x = GP.X_train_
        dkdx =  np.dot(-((xstar - x) / L1**2).T, Kstar1*M1) + \
                np.dot(-((xstar - x) / L2**2).T, Kstar2*M2)
        return np.squeeze(dkdx)

class WindAwarePlanner(LoonPathPlanner):
    def plan(self, loon, pstar):
        z_test = np.linspace(10000, 30000, 100)
        theta = self.__wind_dir__(z_test)
        phi = self.__desired_dir__(loon, pstar)
        candidates = self.__smooth__(phi, theta)
        target = self.__min_climb__(loon, z_test, candidates)
        print("Target altitude: " + str(target))
        print("Direction at target: " + str(theta[z_test == target] * 180.0 / np.pi))
        print("Desired direction: " + str(phi * 180.0 / np.pi - 180.0))
        return target

    def __smooth__(self, phi, theta):
        candidates = np.zeros(len(theta))
        for i in range(len(candidates)):
            avg_candidate = np.cos(phi - theta[i])
            n = 5
            for j in range(n):
                if i+j < len(theta):
                    avg_candidate += np.cos(phi - theta[i+j])
                if i-j >= 0:
                    avg_candidate += np.cos(phi - theta[i-j])
            avg_candidate = avg_candidate / (2*n+1)
            candidates[i] = avg_candidate
        return candidates

    def __min_climb__(self, loon, z_test, candidates):
        p = loon.get_pos()
        idx = (candidates == np.min(candidates))
        min_climb = np.inf
        min_climb_idx = 0
        for i in range(len(idx)):
            flag = idx[i]
            if flag:
                climb = abs(p[2] - z_test[i])
                if climb < min_climb:
                    min_climb = climb
                    min_climb_idx = i
        return z_test[min_climb_idx]

    def __desired_dir__(self, loon, pstar):
        p = loon.get_pos()
        phi = np.arctan2((p[1] - pstar[1]), (p[0] - pstar[0]))
        return phi

    def __wind_dir__(self, z_test):
        vx_test = self.GPx.predict(np.atleast_2d(np.array(z_test)).T, return_std=False)
        vy_test = self.GPy.predict(np.atleast_2d(np.array(z_test)).T, return_std=False)
        theta = np.arctan2(vy_test, vx_test)
        return theta
