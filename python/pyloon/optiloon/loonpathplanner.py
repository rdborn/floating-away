import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import copy
import itertools

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng, downsize
from pyutils import pyutils
from pyflow.pystreams import VarThresholdIdentifier2 as JSI
from optiloon.fieldestimator import GPFE, KNN1DGP, Multi1DGP
from optiloon.pycosts import J_position as Jp
from optiloon.pycosts import J_velocity as Jv
from optiloon.pycosts import J_acceleration as Ja
from optiloon.pycosts import range_J

class LoonPathPlanner:
    def __init__(self, *args, **kwargs):
        """
        Initialize a general loon path planner object.

        kwarg 'field' is the pyflow.flowfields.FlowField used in this simulation.
        kwarg 'lower' is the lowest allowable altitude.
        kwarg 'upper' is the highest allowable altitude.
        """

        field = parsekw(kwargs, 'field', 0.0) # wind field object
        self.fieldestimator = parsekw(kwargs, 'fieldestimator', 'gpfe')
        self.lower = parsekw(kwargs, 'lower', 0.0)
        self.upper = parsekw(kwargs, 'upper', 100)
        self.sampled_points = []
        self.alts_to_sample = np.array([])
        self.train(field=field)
        self.off_nominal = False

    def add_sample(self, *args, **kwargs):
        p = parsekw(kwargs, 'p', None)
        magnitude = parsekw(kwargs, 'magnitude', 0.0)
        direction = parsekw(kwargs, 'direction', 0.0)
        if np.array(p == None).any():
            print("No point supplied, can't sample")
            return -1
        vx = magnitude * np.cos(direction)
        vy = magnitude * np.sin(direction)
        val = np.array([p[2], vx, vy])
        self.sampled_points.append(val)

    def __parse_field__(self, field):
        vx = np.zeros(len(field.field.keys()))
        vy = np.zeros(len(field.field.keys()))
        coords = np.zeros([len(field.coords.keys()),len(field.coords.values()[0])])

        for i in range(len(field.field.keys())):
            this_key = field.field.keys()[i]
            magnitude, direction = field.field[this_key]
            vx[i] = np.array([magnitude * np.cos(direction)])
            vy[i] = np.array([magnitude * np.sin(direction)])
            coords[i] = np.array(field.coords[this_key])
            if np.int(coords[i][2]/1000) == -1:
                 print( "x: " + str(np.int(coords[i][0]/1000)) + \
                        "\ty: " + str(np.int(coords[i][1]/1000)) + \
                        "\tz: " + str(np.int(coords[i][2]/1000)) + \
                        "\tmag: " + str(np.int(magnitude)) + \
                        "\tdir: " + str(np.int(direction*180.0/np.pi)))
        self.mask = downsize(coords)
        coords = coords[:,self.mask]
        return vx, vy, coords

    def train(self, *args, **kwargs):
        """
        Retrain the path planner using a new flow field.

        kwarg 'field' is the pyflow.flowfields.FlowField used in this simulation.
        """

        field = parsekw(kwargs, 'field', 0.0) # wind field object
        vx, vy, coords = self.__parse_field__(field)

        self.kernel = C(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-3, 1e0)) \
                    + C(1.0, (1e-3, 1e3)) * RBF(15, (1e1, 1e3)) \
                    + WhiteKernel(1, (1,1))
        self.n_restarts_optimizer = 9
        if self.fieldestimator == 'gpfe':
            if len(coords[0]) != 1:
                self.kernel = C(1.0, (1e-3, 1e3)) * RBF(0.5*np.ones(len(coords[0])), (1e-6, 1e0)) \
                            + C(1.0, (1e-3, 1e3)) * RBF(15*np.ones(len(coords[0])), (1e1, 1e6)) \
                            + WhiteKernel(1, (1,1))
            self.vx_estimator = GPFE(kernel=self.kernel, n_restarts_optimizer=self.n_restarts_optimizer)
            self.vy_estimator = GPFE(kernel=self.kernel, n_restarts_optimizer=self.n_restarts_optimizer)
        elif self.fieldestimator == 'knn1dgp':
            self.vx_estimator = KNN1DGP()
            self.vy_estimator = KNN1DGP()
        elif self.fieldestimator == 'multi1dgp':
            self.vx_estimator = Multi1DGP()
            self.vy_estimator = Multi1DGP()
        return self.retrain(field=field)

    def retrain(self, *args, **kwargs):
        """
        Retrain the path planner using a new flow field.

        kwarg 'field' is the pyflow.flowfields.FlowField used in this simulation.
        """

        field = parsekw(kwargs, 'field', 0.0) # wind field object
        p = parsekw(kwargs, 'p', np.zeros(3))
        vx, vy, coords = self.__parse_field__(field)

        print(coords.shape)
        print("\tTraining GPx")
        self.vx_estimator.fit(X=coords,
                            y=vx,
                            kernel=self.kernel,
                            n_restarts_optimizer=self.n_restarts_optimizer,
                            p=p)
        print("\tTraining GPy")
        self.vy_estimator.fit(X=coords,
                            y=vy,
                            kernel=self.kernel,
                            n_restarts_optimizer=self.n_restarts_optimizer,
                            p=p)

        return True

    def predict(self, p):
        """
        Estimate the predicted means and standard deviations of the flow field at the provided point.

        parameter p 3D point at which to make the prediction.
        return mean of flow in x direction
        return mean of flow in y direction
        return standard deviation of flow in x direction
        return standard deviation of flow in y direction
        """

        p_pred = np.atleast_2d(np.array(p)[self.mask])
        p_pred = p_pred.T if (np.shape(p_pred)[0] != 1 or np.shape(p_pred)[1] > 3) else p_pred
        mean_fx, std_fx = self.vx_estimator.predict(p=p_pred, return_std=True)
        mean_fy, std_fy = self.vy_estimator.predict(p=p_pred, return_std=True)
        return mean_fx, mean_fy, std_fx, std_fy

    """ HASN'T BEEN USED FOR A WHILE """
    def ev(self, p):
        """
        Estimate only the expected value of the flow field at the provided point.

        parameter p 3D point at which to make the prediction.
        return mean of flow in x direction
        return mean of flow in y direction
        """
        p_pred = np.atleast_2d(np.array(p)[self.mask])
        p_pred = p_pred.T if (np.shape(p_pred)[0] != 1 or np.shape(p_pred)[1] > 3) else p_pred
        mean_fx = self.vx_estimator.predict(p=p_pred, return_std=False)
        mean_fy = self.vy_estimator.predict(p=p_pred, return_std=False)
        return mean_fx, mean_fy

class NaivePlanner(LoonPathPlanner):
    def __init__(self, *args, **kwargs):
        self.last_sounding_position = np.inf * np.ones(2)
        self.last_sounding_time = -np.inf
        self.threshold = parsekw(kwargs, 'resamplethreshold', 200000)
        self.trusttime = parsekw(kwargs, 'trusttime', 3)
        self.points_to_sample = parsekw(kwargs, 'points_to_sample', None)
        self.sampled_points = []
        self.fieldestimator = 'naive'
        self.upper = parsekw(kwargs, 'upper', 30000.)
        self.lower = parsekw(kwargs, 'lower', 0.)


    def add_sample(self, *args, **kwargs):
        p = parsekw(kwargs, 'p', None)
        magnitude = parsekw(kwargs, 'magnitude', 0.0)
        direction = parsekw(kwargs, 'direction', 0.0)
        if np.array(p == None).any():
            print("No point supplied, can't sample")
            return -1
        vx = magnitude * np.cos(direction)
        vy = magnitude * np.sin(direction)
        val = np.array([p[2], vx, vy])
        self.sampled_points.append(val)

    def retrain(self, *args, **kwargs):
        self.data = np.array(self.sampled_points)
        self.sampled_points = []

    def __resample_criteria__(self, *args, **kwargs):
        loon = parsekw(kwargs, 'loon', None)
        tcurr = parsekw(kwargs, 'tcurr', -1)
        curr_pos = np.array(loon.get_pos()[0:2])
        out_of_range = (np.linalg.norm(curr_pos - self.last_sounding_position) > self.threshold)
        been_too_long = ((tcurr - self.last_sounding_time) / 3600) > self.trusttime
        need_to_resample = out_of_range or been_too_long
        print(self.threshold)
        print(np.linalg.norm(curr_pos - self.last_sounding_position))
        return need_to_resample

    def plan(self, *args, **kwargs):
        pstar = parsekw(kwargs, 'pstar', np.zeros(2))
        loon = parsekw(kwargs, 'loon', None)
        tcurr = parsekw(kwargs, 'tcurr', -1)
        need_to_resample = self.__resample_criteria__(**kwargs)
        if need_to_resample:
            self.last_sounding_time = tcurr
            self.last_sounding_position = np.array(loon.get_pos()[0:2])
            return np.array(self.points_to_sample)
        best_alt, best_i, min_J = self.__best_alt__(**kwargs)
        return np.array([best_alt])

    def __best_alt__(self, *args, **kwargs):
        loon = parsekw(kwargs, 'loon', None)
        Jfun = parsekw(kwargs, 'Jfun', Jv)
        pstar = parsekw(kwargs, 'pstar', np.zeros(2))
        curr_pos = np.array(loon.get_pos()[0:2])
        min_J = np.inf
        best_alt = 0
        best_i = 0
        for i, d in enumerate(self.data):
            alt = d[0]
            p = curr_pos[0:2]
            pdot = d[1:]
            J = Jfun(p=p, pstar=pstar[0:2], pdot=pdot)
            if J < min_J:
                min_J = J
                best_alt = alt
                best_i = i
        return best_alt, best_i, min_J

class MolchanovEtAlPlanner(NaivePlanner):
    def __init__(self, *args, **kwargs):
        field = parsekw(kwargs, 'field', 0.0)
        self.upper = parsekw(kwargs, 'upper', 30000.)
        self.lower = parsekw(kwargs, 'lower', 0.)
        self.alpha_threshold = parsekw(kwargs, 'alpha_threshold', 0.2)
        vx, vy, coords = LoonPathPlanner.__parse_field__(self, field)
        self.__partition__(coords, vx, vy)
        R, self.idx = self.__R_max__(6)
        self.altitudes = self.X.values()[0][self.idx]
        NaivePlanner.__init__(self, trusttime=6, threshold=200000, points_to_sample=self.altitudes)
        self.type = 'molchanov'
        self.off_nominal = False
        self.alpha_prev = np.inf
        self.just_retrained = False
        self.prev_pol = -1

    def retrain(self, *args, **kwargs):
        NaivePlanner.retrain(self, *args, **kwargs)
        self.just_retrained = True

    def __partition__(self, X, vx, vy):
        self.locations = dict()
        self.X = dict()
        self.vx = dict()
        self.vy = dict()
        for i in range(len(X)):
            xcoord = np.int(X[i][0])
            ycoord = np.int(X[i][1])
            zcoord = 0
            p = np.array([xcoord, ycoord, zcoord])
            key = hash3d(p)
            alt = np.int(X[i][2])
            if key in self.X.keys():
                self.X[key] = np.append(self.X[key], alt)
                self.vx[key] = np.append(self.vx[key], vx[i])
                self.vy[key] = np.append(self.vy[key], vy[i])
            else:
                self.X[key] = alt
                self.vx[key] = vx[i]
                self.vy[key] = vy[i]
                self.locations[key] = p
        for key in self.locations.keys():
            idx = np.argsort(self.X[key])
            self.X[key] = self.X[key][idx]
            self.vx[key] = self.vx[key][idx]
            self.vy[key] = self.vy[key][idx]
            idx = np.array(self.X[key] < self.upper, dtype=np.int) \
                    + np.array(self.X[key] > self.lower, dtype=np.int) == 2
            self.X[key] = self.X[key][idx]
            self.vx[key] = self.vx[key][idx]
            self.vy[key] = self.vy[key][idx]

    def __R_max__(self, n):
        key = self.X.keys()[0]
        z = self.X[key]
        combos = itertools.combinations(range(len(z)), n)
        idx_max = np.zeros(n)
        R_min = np.inf
        for idx in combos:
            R_i = self.__R__(np.array(idx))
            if R_i < R_min:
                R_min = R_i
                idx_max = np.array(idx)
                # print("R_min:\t" + str(R_min))
                # print(np.sort(self.X[key][idx_max]))
                # print(np.sort(np.arctan(self.vy[key][idx_max], self.vx[key][idx_max])*180/np.pi))
                # pyutils.breakpoint()
        print("\tBest altitudes:")
        alts = np.sort(self.X[key][idx_max])
        for a in alts:
            print("\t\t" + str(np.int(a)) + " m")
        return R_min, idx_max

    def __R__(self, idx):
        R = 0.
        for key in self.locations.keys():
            z = self.X[key][idx]
            theta = np.arctan2(self.vy[key][idx], self.vx[key][idx])
            theta1, theta2 = self.__theta_max_2__(z, theta)
            # print(theta*180/np.pi)
            fa = self.__fa__(theta1, theta2)
            R += fa
            # print(fa)
        return R

    def __fs__(self, x, a):
        a00 = a[0]
        a11 = a[1]
        a12 = a[2]
        a13 = a[3]
        a21 = a[4]
        a22 = a[5]
        a23 = a[6]
        t1 = a11 / (1. + np.exp(a12 * (x - a13)))
        t2 = a21 / (1. + np.exp(a22 * (x - a23)))
        fs = (t1 + t2) / a00
        return fs

    def __fa__(self, theta1, theta2):
        aa = [1., 1., 4., 1., -2., 0.1, 0.1] # this creates a logistic function that maps as closely as I could manage to the one used in the paper
        del_fs = self.__fs__(2., aa) + self.__fs__(0, aa) # make fa = 0 when vectors are colinear
        fs1 = self.__fs__(theta1 / np.pi, aa)
        fs2 = self.__fs__(theta2 / np.pi, aa)
        fa = fs1 + ((np.pi - theta2) / np.pi) * fs2 - del_fs
        return fa

    def __theta_max_2__(self, z, theta):
        max1_i = np.nan
        max1_j = np.nan
        max2_i = np.nan
        max2_j = np.nan
        theta_max1 = 0.
        theta_max2 = 0.
        for i in range(len(theta)):
            for j in range(len(theta)):
                if j > i:
                    d_theta = abs(theta[i] - theta[j])
                    if d_theta > theta_max1:
                        theta_max1 = d_theta
                        max1_i = i
                        max1_j = j
        for i in range(len(theta)):
            for j in range(len(theta)):
                if j > i:
                    d_theta = abs(theta[i] - theta[j])
                    if d_theta > theta_max2:
                        if (i != max1_i) or (j != max1_j):
                            theta_max2 = d_theta
                            max2_i = i
                            max2_j = j
        # print(str(theta_max1*180/np.pi) + "\t" + str(theta[max1_i]*180/np.pi) + "\t" + str(theta[max1_j]*180/np.pi))
        # print(str(theta_max2*180/np.pi) + "\t" + str(theta[max2_i]*180/np.pi) + "\t" + str(theta[max2_j]*180/np.pi))
        return theta_max1, theta_max2

    def __F_desired__(self, *args, **kwargs):
        p = parsekw(kwargs, 'p', None)
        pstar = parsekw(kwargs, 'pstar', np.zeros(3))
        phi = np.array(pstar[0:2] - p[0:2])
        norm_phi = np.linalg.norm(phi)
        F_desired = phi / norm_phi if norm_phi > 0 else phi
        return F_desired

    def __alpha__(self, F_desired):
        Fdnorm = np.linalg.norm(F_desired)
        Fanorm = np.linalg.norm(self.F_actual)
        Fdhat = F_desired / Fdnorm if Fdnorm != 0 else F_desired
        Fahat = self.F_actual / Fanorm if Fanorm != 0 else self.F_actual
        cos_alpha = np.dot(Fdhat, Fahat)
        return np.arccos(cos_alpha)

    def __need_new_action__(self, alpha):
        need_new_action = abs(alpha - self.alpha_prev) > self.alpha_threshold
        if self.just_retrained:
            self.just_retrained = False
            need_new_action = True
        return need_new_action

    def __best_alt__(self, F_desired):
        best_alt = 0
        best_i = 0
        qmax = -np.inf
        Fdnorm = np.linalg.norm(F_desired)
        Fdhat = F_desired / Fdnorm if Fdnorm != 0 else F_desired
        for i, d in enumerate(self.data):
            alt = d[0]
            F_i = d[1:]
            Finorm = np.linalg.norm(F_i)
            Fihat = F_i / Finorm if Finorm != 0 else F_i
            q = np.dot(Fdhat, Fihat)
            if q > qmax:
                qmax = q
                best_alt = alt
                best_i = i
                # print(d)
                # print(q)
                # print(alt)
                # pyutils.breakpoint()
        return best_alt, best_i

    def plan(self, *args, **kwargs):
        need_to_resample = NaivePlanner.__resample_criteria__(self, **kwargs)
        loon = parsekw(kwargs, 'loon', None)
        if need_to_resample:
            self.last_sounding_position = np.array(loon.get_pos()[0:2])
            self.last_sounding_time = parsekw(kwargs, 'tcurr', None)
            pol = self.points_to_sample
        else:
            F_desired = self.__F_desired__(p=loon.get_pos(), pstar=kwargs.get('pstar'))
            self.F_actual = np.array(loon.get_vel()[0:2])
            alpha = self.__alpha__(F_desired)
            need_new_action = self.__need_new_action__(alpha)
            print(alpha)
            print(self.alpha_prev)
            print(self.alpha_threshold)
            if need_new_action:
                pol, i = self.__best_alt__(F_desired)
                if pol == self.prev_pol:
                    pol = -1
                else:
                    self.prev_pol = pol
                self.F_actual = np.array(self.data[i][1:])
                self.alpha_prev = self.__alpha__(F_desired)
            else:
                pol = -1
        pol = np.array(pol)
        if len(pol.shape) == 0:
            pol = np.array([pol])
        return pol

""" HASN'T BEEN USED FOR A WHILE """
class MonteCarloPlanner(LoonPathPlanner):
    def __init__(self, *args, **kwargs):
        """
        Initialize a loon path planner that uses Monte Carlo tree search to plan.

        kwarg 'field' is the pyflow.flowfields.FlowField used in this simulation.
        kwarg 'lower' is the lowest allowable altitude.
        kwarg 'upper' is the highest allowable altitude.
        """

        self.nodes = dict()
        self.edges = dict()
        self.backedges = dict()
        self.values = dict()
        self.leaves = np.array([[np.inf, np.inf, np.inf, np.inf]])
        self.branch_length = 180 # seconds
        LoonPathPlanner.__init__(   self,
                                    field=kwargs.get('field'),
                                    lower=kwargs.get('lower'),
                                    upper=kwargs.get('upper'))

    """ HASN'T BEEN USED FOR A WHILE """
    def __reset__(self):
        """
        Clears all sets associated with the Monte Carlo tree search.

        return success/failure
        """
        self.nodes = dict()
        self.edges = dict()
        self.backedges = dict()
        self.values = dict()
        self.leaves = np.array([[np.inf, np.inf, np.inf, np.inf]])
        return True

    def plan(self, *args, **kwargs):
        """
        Plan the path for the balloon.

        parameter loon balloon object for which to plan.
        parameter pstar 3D point, goal location
        parameter depth recursion depth for Monte Carlo tree search
        return optimal policy (discrete time control effort in reverse order)
        """
        loon = parsekw(kwargs, 'loon', None)
        pstar = parsekw(kwargs, 'pstar', np.zeros(3))
        depth = parsekw(kwargs, 'depth', 0)

        self.__reset__()
        self.__montecarlo__(loon, pstar, depth)
        return self.__policy__()

    def __montecarlo__(self, loon, pstar, depth):
        """
        Execute Monte Carlo tree search for the balloon.

        parameter loon balloon object for which to plan.
        parameter pstar 3D point, goal location
        parameter depth recursion depth for Monte Carlo tree search
        """

        # Add this node and get its value
        x, y, z = loon.get_pos()
        self.__add_node__([x,y,z], loon)

        # Base case
        if depth <= 0:
            self.__add_leaf__([x,y,z])
            return

        u = 5
        self.__climb_branch__(loon, u, pstar, depth-1)
        self.__climb_branch__(loon, 0, pstar, depth-1)
        self.__climb_branch__(loon, -u, pstar, depth-1)

    def __add_node__(self, p, loon):
        """
        Add a node to the tree built using Monte Carlo tree search.

        parameter p 3D point, node to add.
        parameter loon balloon object, node value.
        return success/failure
        """

        self.nodes[hash3d(p)] = copy.deepcopy(loon)
        if self.__add_cost__(p, pstar):
            return True
        return False

    def __add_cost__(self, p, pstar):
        """
        Calculate and store integrated cost from root to node.

        parameter p 3D point, node for which to calculated integrated cost.
        parameter pstar 3D point, goal location
        return success/failure
        """

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
        """
        Record the fact that this node is a leaf.

        parameter p node to add as a leaf.
        return success/failure
        """

        val = self.values[hash3d(p)]
        self.leaves = np.append(self.leaves, [[p[0], p[1], p[2], val]], axis=0)
        return True

    def __climb_branch__(self, loon, u, pstar, depth):
        """
        Propogate balloon dynamics along a branch in the Monte Carlo tree search.

        parameter loon balloon object to propogate.
        parameter u control effort (verticaly velocity) to apply to loon.
        parameter pstar 3D point, goal location.
        parameter depth recursion depth for Monte Carlo tree search
        """

        working_loon = copy.deepcopy(loon)
        w = 0
        c_noise = 1e-5
        for i in range(int(np.ceil(self.branch_length * working_loon.Fs))):
            # Get expected value of wind at each propogating position
            # NOTE: terminal velocity is assumed at every time step
            #       (i.e. drag force reaches zero arbitrarily fast)
            expected_value = np.squeeze(self.ev(working_loon.get_pos()))
            vwind_x = expected_value[0] * np.cos(expected_value[1])
            vwind_y = expected_value[0] * np.sin(expected_value[1])
            lp = working_loon.update(   vz=u + rng(1e-3),
                                        vx=vwind_x,
                                        vy=vwind_y)

            # Calculate weights of these edges
            w += self.__cost__(lp, pstar)

        x, y, z = loon.get_pos()
        wlx, wly, wlz = working_loon.get_pos()
        self.edges[hash4d([x,y,z,u])] = [hash3d([wlx, wly, wlz]), w]
        self.backedges[hash3d([wlx,wly,wlz])] = [hash3d([x,y,z]), u, w]

        self.__montecarlo__(working_loon, pstar, depth)

    def __policy__(self):
        """
        Return optimal policy for most recent planning loop.

        return optimal policy (discrete time control effort in reverse order).
        """

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

""" HASN'T BEEN USED FOR A WHILE """
class PlantInvertingController(LoonPathPlanner):
    def __init__(self, *args, **kwargs):
        LoonPathPlanner.__init__(self, *args, **kwargs)
        self.GPx = self.vx_estimator.get_current_estimator()
        K1x, K2x = self.__recreate_gp__(self.GPx)
        self.xx = self.GPx.X_train_
        yx = self.GPx.y_train_
        self.M1x = np.dot(np.linalg.inv(K1x),yx)
        self.M2x = np.dot(np.linalg.inv(K2x),yx)
        thetax = np.exp(self.GPx.kernel_.theta)
        self.L1x = thetax[1]
        self.L2x = thetax[3]

        self.GPy = self.vy_estimator.get_current_estimator()
        K1y, K2y = self.__recreate_gp__(self.GPy)
        self.xy = self.GPy.X_train_
        yy = self.GPy.y_train_
        self.M1y = np.dot(np.linalg.inv(K1y),yy)
        self.M2y = np.dot(np.linalg.inv(K2y),yy)
        thetay = np.exp(self.GPy.kernel_.theta)
        self.L1y = thetay[1]
        self.L2y = thetay[3]

    def retrain(self, *args, **kwargs):
        LoonPathPlanner.retrain(self, *args, **kwargs)
        self.GPx = self.vx_estimator.get_current_estimator()
        K1x, K2x = self.__recreate_gp__(self.GPx)
        self.xx = self.GPx.X_train_
        yx = self.GPx.y_train_
        self.M1x = np.dot(np.linalg.inv(K1x),yx)
        self.M2x = np.dot(np.linalg.inv(K2x),yx)
        thetax = np.exp(self.GPx.kernel_.theta)
        self.L1x = thetax[1]
        self.L2x = thetax[3]

        self.GPy = self.vy_estimator.get_current_estimator()
        K1y, K2y = self.__recreate_gp__(self.GPy)
        self.xy = self.GPy.X_train_
        yy = self.GPy.y_train_
        self.M1y = np.dot(np.linalg.inv(K1y),yy)
        self.M2y = np.dot(np.linalg.inv(K2y),yy)
        thetay = np.exp(self.GPy.kernel_.theta)
        self.L1y = thetay[1]
        self.L2y = thetay[3]

    def plan(self, *args, **kwargs):
        """
        Calculate control effort to accelerate towards the origin.

        kwarg loon balloon object for which to plan
        return plant-inverting control effort (i.e. attempt to accelerate towards origin)
        """

        loon = parsekw(kwargs, 'loon', None)
        retrain = parsekw(kwargs, 'retrain', False)
        if retrain:
            self.retrain()
        p = np.array(loon.get_pos())
        if loon == None:
            warning("Must specify loon.")
        J_x, J_y = self.__dkdx__(p[2])
        J = np.array([J_x, J_y])
        print(J)
        x = p[0:2]
        if np.inner(x,x) == 0:
            return 5.
        num = np.inner(J, -x / np.inner(x, x))
        den = np.inner(J,J)
        u = num / den
        # print("u: " + str(u))
        # u = pyutils.saturate(u, 5)
        if abs(u) < 1e-3:
            u = 0
        elif u > 0:
            u = 5.
        else:
            u = -5.
        if u != 0:
            print("u: " + str(u))
        return u

    def __kstar_gp__(self, GP, xstar):
        """
        Back out test covariance matrix for Gaussian Process estimation.

        parameter GP Gaussian Process from which to back out test covariance matrix.
        parameter xstar point at which to calculate test covariance matrix.
        return test covariance matrix (or matrices for compound kernels)
        """
        # NOTE: Cannot distinguish between different kernel types
        #       (i.e. function is hardcoded for two summed RBF kernels)

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
        """
        Back out training covariance matrix for Gaussian Processes estimation.

        parameter GP Gaussian Process from which to back out training covariance matrix.
        return training covariance matrix (or matrices for compound kernels)
        """
        # NOTE: Cannot distinguish between different kernel types
        #       (i.e. function is hardcoded for two summed RBF kernels)

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
        # M = np.dot(np.linalg.inv(K1) + np.linalg.inv(K2) + np.eye(len(x))*sigma, y)
        return K1, K2

    def __dkdx__(self, xstar):
        """
        Differentiate a Gaussian Process.

        parameter xstar test point at which to differentiate.
        return derivative of GP at xstar
        """
        # NOTE: Cannot distinguish between different kernel types
        #       (i.e. function is hardcoded for two summed RBF kernels)

        Kstar1x, Kstar2x = self.__kstar_gp__(self.GPx, xstar)
        Kstar1y, Kstar2y = self.__kstar_gp__(self.GPy, xstar)
        dkdx_x =  np.dot(-((xstar - self.xx) / self.L1x**2).T, Kstar1x*self.M1x) + \
                np.dot(-((xstar - self.xx) / self.L2x**2).T, Kstar2x*self.M2x)
        dkdx_y =  np.dot(-((xstar - self.xy) / self.L1y**2).T, Kstar1y*self.M1y) + \
                np.dot(-((xstar - self.xy) / self.L2y**2).T, Kstar2y*self.M2y)
        dkdx_x = np.squeeze(dkdx_x)
        dkdx_y = np.squeeze(dkdx_y)
        return dkdx_x, dkdx_y

class WindAwarePlanner(LoonPathPlanner):
    def __init__(self, *args, **kwargs):
        """
        Initialize a wind-aware path planner for a balloon.

        kwarg 'field' pyflow.flowfields.FlowField used in this simulation.
        kwarg 'lower' lowest allowable altitude.
        kwarg 'upper' highest allowable altitude.
        kwarg 'streamres' number of points to sample from air column.
        kwarg 'streammin' minimum altitude from which to sample.
        kwarg 'streammax' maximum altitude from which to sample.
        kwarg 'threshold' variance threshold for identifying unique jetstreams.
        kwarg 'streamsize' minimum number of sample points for a jetstream to be classified as such.
        """

        LoonPathPlanner.__init__(   self,
                                    field=kwargs.get('field'),
                                    lower=kwargs.get('lower'),
                                    upper=kwargs.get('upper'),
                                    fieldestimator=kwargs.get('fieldestimator'))
        self.streamres = parsekw(kwargs, 'streamres', 500)
        self.streammax = parsekw(kwargs, 'streammax', 30000)
        self.streammin = parsekw(kwargs, 'streammin', 0)
        self.threshold = parsekw(kwargs, 'threshold', 0.01)
        self.streamsize = parsekw(kwargs, 'streamsize', 20)
        self.__redo_jetstreams__(np.zeros(3))

    def __redo_jetstreams__(self, p):
        alt = np.linspace(self.streammin, self.streammax, self.streamres)
        # vx = np.zeros(self.streamres)
        # vy = np.zeros(self.streamres)
        # stdx = np.zeros(self.streamres)
        # stdy = np.zeros(self.streamres)
        p_test = np.ones([self.streamres,3])
        p_test[:,0] = p[0]
        p_test[:,1] = p[1]
        p_test[:,2] = alt
        vx, vy, stdx, stdy = self.predict(p_test.T)
        # for i, z in enumerate(alt):
            # vx_i, vy_i, std_x_i, std_y_i = self.predict(np.array([p[0], p[1], z]))
            # vx[i] = vx_i
            # vy[i] = vy_i
            # stdx[i] = std_x_i
            # stdy[i] = std_y_i
        self.jets = JSI(vx=vx,
                        vy=vy,
                        stdx=stdx,
                        stdy=stdy,
                        alt=alt,
                        threshold=self.threshold,
                        streamsize=self.streamsize,
                        expectation=False)
        self.jets_expectation = JSI(vx=vx,
                                    vy=vy,
                                    stdx=stdx,
                                    stdy=stdy,
                                    alt=alt,
                                    threshold=self.threshold,
                                    streamsize=self.streamsize,
                                    expectation=True)
        print(self.jets)
        print(self.jets_expectation)

    def plan(self, *args, **kwargs):
        """
        Calculate and return which jetstream for a balloon to travel to.

        parameter loon balloon for which to plan.
        parameter pstar 3D point, goal location.
        return altitude of center of most desirable jetstream.
        """
        loon = parsekw(kwargs, 'loon', None)
        pstar = parsekw(kwargs, 'pstar', np.zeros(3))

        self.__redo_jetstreams__(loon.get_pos())
        print(self.jets.jetstreams.values())

        phi = self.__desired_dir__(loon, pstar)
        target, J = self.__find_best_jetstream__(loon, pstar, 5)
        print("Target altitude: " + str(target.avg_alt))
        print("Direction at target: " + str((target.direction * 180.0 / np.pi) % 360))
        print("Desired direction: " + str((phi * 180.0 / np.pi - 180.0) % 360))
        return target.avg_alt

    """ NOT SUPPORTED """
    def __smooth__(self, phi, theta):
        """
        NOT SUPPORTED
        Smooth the data and return candidates for use in planning.

        parameter phi desired direction
        parameter theta unsmoothed data, wind direction at various altitudes
        """

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

    """ HASN'T BEEN USED FOR A WHILE """
    def __min_climb__(self, loon, z_test, candidates):
        """
        If multiple jetstreams are equally desirable, find the closer one.

        parameter loon balloon for which to plan.
        parameter z_test candidate altitudes.
        parameter candidates (negative) desirability of each altitude in z_test.
        """

        p = loon.get_pos()
        idx = (candidates == np.min(candidates))
        min_climb = np.inf
        min_climb_idx = 0
        for i in range(len(idx)):
            flag = idx[i] and z_test[i] > self.lower and z_test[i] < self.upper
            if flag:
                climb = abs(p[2] - z_test[i])
                if climb < min_climb:
                    min_climb = climb
                    min_climb_idx = i
        return z_test[min_climb_idx]

    """ HASN'T BEEN USED FOR A WHILE """
    def __desired_dir__(self, loon, pstar):
        """
        Calculate desired direction for balloon to travel.

        parameter loon balloon for which to plan.
        parameter pstar 3D point, goal location.
        return direction from loon's current position directly to pstar
        """

        p = loon.get_pos()
        phi = np.arctan2((p[1] - pstar[1]), (p[0] - pstar[0]))
        return phi

    """ HASN'T BEEN USED FOR A WHILE """
    def __wind_dir__(self, z_test):
        """
        Calculate wind direction at various altitudes.

        parameter z_test altitudes at which to calculate (estimated) wind direction.
        return estimated wind directions at altitudes in z_test
        """

        vx_test = self.vx_estimator.predict(p=np.array(z_test), return_std=False)
        vy_test = self.vy_estimator.predict(p=np.array(z_test), return_std=False)
        theta = np.arctan2(vy_test, vx_test)
        return theta

    """ HASN'T BEEN USED FOR A WHILE """
    def __find_best_jetstream__(self, loon, pstar, u):
        """
        Consider each jetstream and choose which one would be best to travel to.

        parameter loon balloon object on which to plan
        parameter pstar goal position
        parameter u control effort (i.e. vertical velocity magnitude)
        return lowest cost jetstream and its cost
        """

        # SETTING UP/INITIALIZATIONS:
        # Get the number of jetstreams
        n_streams = len(self.jets.jetstreams)
        # Initialize the terminal cost of each jetstream to inf
        J_jetstreams = np.inf * np.ones(n_streams)
        # Initialize the fuel cost to travel to each jetstream to inf
        J_fuel_jetstreams = np.inf * np.ones(n_streams)
        # Initialize the total cost of each jetstream to inf
        J = np.inf * np.ones(n_streams)
        # Get the balloon's position
        pos = loon.get_pos()
        x_loon = pos[0]
        y_loon = pos[1]
        # Initialize the altitudes of each jetstream to zero
        z_jets = np.zeros(n_streams)
        # Arbitrarily initialize the best jetstream to the first jetstream
        best_jet = self.jets.jetstreams.values()[0]
        # Initialize the total cost of the best jetstream to inf
        best_J = np.inf

        # SEARCHING FOR THE JETSTREAM WITH THE LOWEST TOTAL COST:
        # For each jetstream...
        for i, jet in enumerate(self.jets.jetstreams.values()):
            # Store the jetstream's altitude
            z_jets[i] = jet.avg_alt
            # Find the change in lateral and vertical position to travel to this
            # this jetstream
            dp, dz = self.__cost_to_altitude__(loon, z_jets[i], u)
            # Find the new position the balloon would be at after traveling to
            # this jetstream
            dx = dp[0]
            dy = dp[1]
            new_pos = np.array([dx, dy, dz]) + pos
            # Calculate the fuel cost of traveling to this jetstream as the
            # vertical distance to it (scaled)
            J_fuel_jetstreams[i] = dz*1e-3
            # Calculate the terminal cost of this jetstream
            J_jetstreams[i] = Jv(p=new_pos[0:2], pstar=pstar, pdot=jet.v)
            # Calculate the total cost of choosing this jetstream as the sum
            # of the fuel cost and terminal cost
            J[i] = J_fuel_jetstreams[i] + J_jetstreams[i]
            # If this is the best jetstream we've considered so far...
            if J[i] < best_J:
                # Update the best cost and best jetstream variables appropriately
                best_J = J[i]
                best_jet = jet
            print("Alt: " + str(jet.avg_alt) + "\t\tJf: " + str(np.int(J_fuel_jetstreams[i])) + "\t\tJj: " + str(np.int(J_jetstreams[i])))

        # Return the jetstream with the smallest total cost and its cost
        return best_jet, best_J

    """ HASN'T BEEN USED FOR A WHILE """
    def __moving_towards_target__(self, pos, pstar, jet):
        p = pos[0:2]
        pstar = pstar[0:2]
        magnitude = jet.magnitude
        direction = jet.direction
        vx = magnitude * np.cos(direction)
        vy = magnitude * np.sin(direction)
        pdot = np.squeeze(np.array([vx, vy]).T)
        pdothat = pdot / np.linalg.norm(pdot)
        phi = p - pstar
        norm_phi = np.linalg.norm(phi)
        phihat = phi / norm_phi if norm_phi > 0 else phi
        moving_towards = (np.dot(phihat, pdothat) < 0)
        return moving_towards

    """ HASN'T BEEN USED FOR A WHILE """
    def __cost_to_altitude__(self, loon, z, u):
        """
        Calculate the lateral and vertical displacements incurred during a
        transit from the balloon's current position to the given target altitude.

        parameter loon balloon for which to calculate displacements
        parameter z target altitude
        parameter u control effort during transit (constant)
        return lateral and vertical displacement incurred during transit
        """

        # SETTING UP/INITIALIZATION
        # Get the balloon's current position
        pos = loon.get_pos()
        z_loon = pos[2]
        # Generate test points from loon's current position to target altitude
        N = 500
        x_test = np.zeros(N)
        y_test = np.zeros(N)
        z_test = np.linspace(z_loon, z, N)
        p_test = np.array([x_test, y_test, z_test])
        # Get wind velocities at each test point
        vx, vy = self.ev(p_test)
        # Find the mean wind velocity across all test points
        mean_vx = np.mean(vx)
        mean_vy = np.mean(vy)

        # CALCULATE DISPLACEMENTS
        # Find the change in altitude from the loon's current position to
        # target altitude
        dz = abs(z - z_loon)
        # Find the time required to travel to the target altitude given
        # control effort u
        t = dz / u
        # Find the change in lateral position incurred during the transit to
        # the target altitude
        dx = mean_vx * t
        dy = mean_vy * t
        dp = np.array([dx, dy])

        # Return the lateral and vertical displacement to travel from the loon's
        # current position to the target altitude
        return dp, dz

""" HASN'T BEEN USED FOR A WHILE """
class MPCWAP(WindAwarePlanner):
    def plan(self, *args, **kwargs):
        """
        Find the best sequence of jetstreams to which to travel. Balloon can
        only move to adjacent jetstreams.

        parameter loon balloon for which to plan
        parameter u control effort (constant during transit between jetstreams)
        parameter T length of time to stay at current jetstream for the 'stay' option
        parameter pstar set point/goal position
        parameter depth length of jetstream sequences for consideration
        return sequence of jetstreams that has the smallest total cost
        """
        loon = parsekw(kwargs, 'loon', None)
        u = parsekw(kwargs, 'u', 0.0)
        T = parsekw(kwargs, 'T', 180.0)
        pstar = parsekw(kwargs, 'pstar', np.zeros(3))
        depth = parsekw(kwargs, 'depth', 0)

        # Initialize a dictionary to store the potential sequences of jetstreams,
        # indexed by their cost
        self.sequences = dict()
        # Conduct a recursive tree search to populate the sequences dictionary
        self.__tree_search__(loon, u, T, pstar, depth, 0.0, np.array([]))
        # Find the minimum cost among the possible sequences
        min_J = np.min(self.sequences.keys())
        # Find the sequence/policy associated with the minimum cost
        best_pol = self.sequences[min_J]
        # Return the minimum cost policy
        return best_pol

    def __tree_search__(self, loon, u, T, pstar, depth, J, policy):
        """
        Recursively find the cost of every possible sequence of jetstreams of
        a given length. Balloon can only move to adjacent jetstreams.

        parameter loon balloon for which to plan
        parameter u control effort (constant during transit between jetstreams)
        parameter T length of time to stay at current jetstream for the 'stay' option
        parameter pstar set point/goal position
        parameter depth length of jetstream sequences for consideration
        """
        # Base case
        if depth == 0:
            # Index this policy by its cost and store it
            self.sequences[J] = policy
            # Do nothing else
            return

        # Find the cost of moving up, down, or staying put
        up_cost, up_loon = self.__cost_2_move__(loon, u, pstar)
        down_cost, down_loon = self.__cost_2_move__(loon, -u, pstar)
        stay_cost, stay_loon = self.__cost_2_stay__(loon, T, pstar)

        # Get jetstream loon will move to if it goes up, down, or stays put
        up_jet = self.jets.find(up_loon.get_pos()[2])
        down_jet = self.jets.find(down_loon.get_pos()[2])
        stay_jet = self.jets.find(stay_loon.get_pos()[2])

        # Append policy-so-far with moving up, down, or staying put
        up_policy = np.append(np.array(policy), np.array([up_jet.avg_alt]))
        down_policy = np.append(np.array(policy), np.array([down_jet.avg_alt]))
        stay_policy = np.append(np.array(policy), np.array([stay_jet.avg_alt]))

        # Recursive calls
        self.__tree_search__(up_loon, u, T, pstar, depth-1, J+up_cost, up_policy)
        self.__tree_search__(down_loon, u, T, pstar, depth-1, J+down_cost, down_policy)
        self.__tree_search__(stay_loon, u, T, pstar, depth-1, J+stay_cost, stay_policy)

    def __cost_2_move__(self, loon, u, pstar):
        """
        Calculate cost associated with moving to an adjacent jetstream, whether
        to move up or down is indicated by the sign of the given control effort.

        parameter loon balloon for which to calculate cost
        parameter u control effort (constant during transit)
        parameter pstar set point/goal position
        return cost of moving to adjacent jetstream and propogated loon
        """

        # SETTING UP/INITIALIZATION
        # Get balloon's current position
        pos = loon.get_pos()
        z = pos[2]
        # Get the jet stream the balloon is in currently and the one directly
        # above or below, depending on the given control effort
        curr_jetstream = self.jets.find(z)
        next_jetstream = self.jets.find_adjacent(z, u)
        # Get the altitude of the adjacent jetstream
        target_alt = next_jetstream.avg_alt
        # Copy the balloon so we can modify it without fear
        test_loon = copy.deepcopy(loon)
        # If the find_adjacent() function returned the current jetstream
        # instead of the adjacent one, that means we are at the edge of the air
        # column and there is no jetstream adjacent to us. Return a cost of inf
        if target_alt == curr_jetstream.avg_alt:
            print("No jetstream in that direction. Returning cost of inf")
            return np.inf, test_loon

        # FIND COST TO MOVE TO ADJACENT JETSTREAM
        # Get the copied loon's position
        test_pos = test_loon.get_pos()
        # Initialize the cost and simulation time to 0
        J_pos = 0.0
        dt = 0.0
        # While we have not reached the next jetstream...
        while (target_alt - test_pos[2]) * np.sign(u) > 0:
            # Get the wind velocity at the test loon's current position
            vx, vy = self.ev(test_pos)
            # Propogate the test loon's dynamisc
            test_loon.update(vx=vx, vy=vy, vz=u)
            # Update the positon of the test loon
            test_pos = test_loon.get_pos()
            # Get the new displacement from the set point
            dp = pstar - test_pos
            # Calculate the cost associated with the current displacement from
            # the set point and add it to the integrated cost
            J_pos += np.sum((dp[0:2])**2)
            # Update the simulation time
            dt += 1.0 / loon.Fs
        # Update the integrated cost to the the root-mean-squared cost over the
        # course of the simulation.
        J_pos = np.sqrt(J_pos / dt)
        # Calculate the terminal cost of arriving at this jetstream
        J_vel = WindAwarePlanner.__cost_of_jetstream__(self, test_pos, pstar, next_jetstream)
        # Calculate the total cost of moving to this jetstream as the sum of the
        # root-mean-square position cost, terminal cost, and cost of fuel to
        # move to this jetstream
        J = (J_pos + J_vel + (target_alt - z)**2*1e-5)

        # Return the total cost and the propogated test loon
        return J, test_loon

    def __cost_2_stay__(self, loon, T, pstar):
        pos = loon.get_pos()
        z = pos[2]
        curr_jetstream = self.jets.find(z)
        test_loon = copy.deepcopy(loon)
        test_pos = test_loon.get_pos()
        vx, vy = self.ev(test_pos)
        J_pos = 0
        dt = 0
        while (T - dt) > 0:
            test_loon.update(vx=vx, vy=vy)
            test_pos = test_loon.get_pos()
            dp = pstar - test_pos
            J_pos += np.sum((dp[0:2])**2)
            dt += 1.0 / loon.Fs
        J_pos = np.sqrt(J_pos / dt)
        J_vel = WindAwarePlanner.__cost_of_jetstream__(self, test_pos, pstar, curr_jetstream)
        J = (J_pos + J_vel)
        return J, test_loon

class MPCWAPFast(WindAwarePlanner):
    def __init__(self, *args, **kwargs):
        WindAwarePlanner.__init__(self, **kwargs)
        u = parsekw(kwargs, 'u', 5.0)
        self.off_nominal = True
        print("nominal dp")
        self.__delta_p_between_jetstreams__(u)
        self.off_nominal = False
        print("off nominal dp")
        self.__delta_p_between_jetstreams__(u)
        self.__find_altitudes_for_sampling__(100, 0.3)
        self.current_estimator = self.vx_estimator.prediction_key
        self.type = 'mpcfast'

    def __redo_jetstreams_etc__(self, u, loon):
        WindAwarePlanner.__redo_jetstreams__(self, loon.get_pos())
        self.off_nominal = not self.off_nominal
        self.__delta_p_between_jetstreams__(u)
        self.off_nominal = not self.off_nominal
        self.__delta_p_between_jetstreams__(u)
        self.__find_altitudes_for_sampling__(100, 0.3)

    def __incorporate_samples__(self, redo_stuff, u, loon):
        if len(self.sampled_points) > 0:
            print("Omg new data!")
            new_X = np.array(self.sampled_points)[:,0]
            new_y_x = np.array(self.sampled_points)[:,1]
            new_y_y = np.array(self.sampled_points)[:,2]
            self.vx_estimator.add_data(X=new_X, y=new_y_x)
            self.vy_estimator.add_data(X=new_X, y=new_y_y)
            self.sampled_points = []
            if redo_stuff:
                self.__redo_jetstreams_etc__(u, loon)

    def __reset_plan__(self, p):
        self.backedges = dict()
        self.curr_key = 0
        self.leaves = dict()
        self.lowest_J_yet = np.inf
        self.nodes_expanded = 0
        trash, trash, trash, trash = self.predict(p)
        self.current_estimator = self.vx_estimator.prediction_key

    def __preplan__(self, loon, u, radius):
        new_estimator = self.vx_estimator.prediction_key
        if self.current_estimator != new_estimator:
            self.__incorporate_samples__(False, u, loon)
            self.__redo_jetstreams_etc__(u, loon)
        else:
            self.__incorporate_samples__(True, u, loon)
        # print(self.jets)
        self.__reset_plan__(loon.get_pos())

    def __check_nominality__(self, threshold):
        estimate_quality = self.jets.spanning_quality()
        expected_quality = self.jets_expectation.spanning_quality()
        delta_quality = estimate_quality - expected_quality
        self.off_nominal = delta_quality > threshold

    def plan(self, *args, **kwargs):
        loon = parsekw(kwargs, 'loon', None)
        u = parsekw(kwargs, 'u', 0.0)
        T = parsekw(kwargs, 'T', 180.0)
        pstar = parsekw(kwargs, 'pstar', np.zeros(3))
        depth = parsekw(kwargs, 'depth', 0)
        gamma = parsekw(kwargs, 'gamma', 1e7)

        radius = 40000
        delta_quality_threshold = 0.1
        print("preplan")
        self.__preplan__(loon, u, radius)
        self.__check_nominality__(delta_quality_threshold)
        print("tree search")
        self.__tree_search__(loon.get_pos(), 1, u, T, pstar, depth, 0.0, np.array([]), gamma)
        best_pol, best_J = self.__best_pol__(p=loon.get_pos())
        return best_pol

    def __sample_destination__(self, p, jet):
        alts = np.linspace(jet.min_alt, jet.max_alt, 100)
        max_H = -np.inf
        chosen_alt = None
        for alt in alts:
            p_test = np.array([p[0], p[1], alt])
            vx, vy, stdx, stdy = self.predict(p_test)
            H = pyutils.bivar_normal_entropy(stdx, stdy)
            if H > max_H:
                max_H = H
                chosen_alt = alt
        print(chosen_alt)
        return np.array([chosen_alt, max_H])

    def __get_stay_branch_length__(self, pos, pstar, jet, T, this_id):
        dp = np.zeros(2)
        prev_accel_cost = Ja(p=pos[0:2], pstar=pstar, pdot=jet.v)
        new_accel_cost = Ja(p=pos[0:2]+dp, pstar=pstar, pdot=jet.v)
        d_accel_cost = abs(new_accel_cost - prev_accel_cost)
        magnitude = jet.magnitude
        direction = jet.direction
        vx = magnitude * np.cos(direction)
        vy = magnitude * np.sin(direction)
        total_T = 0.0
        if WindAwarePlanner.__moving_towards_target__(self, pos[0:2], pstar, jet):
            while d_accel_cost < 0.5*prev_accel_cost:
                total_T += T
                dp = np.array([vx, vy]) * total_T
                new_accel_cost = Ja(p=pos[0:2]+dp, pstar=pstar, pdot=jet.v)
                d_accel_cost = abs(new_accel_cost - prev_accel_cost)
        else:
            total_T += T
            dp = np.array([vx, vy]) * total_T
        if self.off_nominal:
            dp_std = self.delta_std_expectation[jet.id,this_id] * total_T
        else:
            dp_std = self.delta_std[jet.id,this_id] * total_T
        return dp, dp_std

    def __J_sample__(self, pos, new_pos, pstar, target_alt):
        J_sample = 0.0
        for potential_sample in self.alts_to_sample:
            on_the_way = (potential_sample > pos[2] and potential_sample < target_alt) or \
                        (potential_sample < pos[2] and potential_sample > target_alt)
            if on_the_way:
                p_sample = np.array([new_pos[0], new_pos[1], potential_sample])
                vx_samp, vy_samp, stdx_samp, stdy_samp = self.predict(p_sample)
                if self.__cost_of_vel__(p_sample[0:2], pstar, np.array([vx_samp, vy_samp])) < 1:
                    # print("Omg we get to sample")
                    # print(str((stdx_samp**2 + stdy_samp**2)))
                    J_sample += -(stdx_samp**2 + stdy_samp**2)
        return J_sample

    """ HACK """
    def __hacky_way_to_avoid_bug_when_not_in_jetstream__(self, pos, jet_id):
        buf = 10.0
        while jet_id < 0:
            buf += np.sign(buf) * 10.0
            buf *= -1
            if self.off_nominal:
                this_jet = self.jets_expectation.find(pos[2] + buf)
            else:
                this_jet = self.jets.find(pos[2] + buf)
            jet_id = this_jet.id
            # print(buf)
        if buf > 10:
            print(buf)
        return jet_id

    def __dp__(self, p, pstar, T, this_id, jet):
        if jet.id == this_id:
            dp, dp_std = self.__get_stay_branch_length__(p, pstar, jet, T, this_id)
        else:
            if self.off_nominal:
                dp = self.delta_p_expectation[jet.id,this_id]
                dp_std = self.delta_std_expectation[jet.id,this_id]
            else:
                dp = self.delta_p[jet.id,this_id]
                dp_std = self.delta_std[jet.id,this_id]
        return dp, dp_std

    def __terminate_branch__(self, key, val):
        self.leaves[key] = val
        J = val[2]
        if J < self.lowest_J_yet:
            self.lowest_J_yet = J

    def __tree_search__(self, pos, prev_key, u, T, pstar, depth, J, policy, gamma):
        self.curr_key += 1
        this_key = self.curr_key
        back_key = prev_key
        val = np.array([back_key, pos[2], J, pos[0], pos[1]])
        self.backedges[this_key] = val

        self.nodes_expanded += 1

        if depth == 0:
            self.__terminate_branch__(this_key, val)
            return

        if self.off_nominal:
            # print("Off nominal")
            jets = self.jets_expectation
        else:
            jets = self.jets
        this_jet = jets.find(pos[2])
        this_id = this_jet.id
        this_id = self.__hacky_way_to_avoid_bug_when_not_in_jetstream__(pos, this_id)
        for i, jet in enumerate(jets.jetstreams.values()):
            if jet.avg_alt > self.lower and jet.avg_alt < self.upper:
                target_alt = jet.avg_alt
                dp, dp_std = self.__dp__(pos, pstar, T, this_id, jet)
                new_pos = np.append(pos[0:2] + dp, target_alt)
                gamma = gamma * np.ones(6) if len(gamma) == 1 else gamma
                p = new_pos[0:2]

                if abs(gamma[2]) < 1e-6:
                    J_pos = Jp(p=p, pstar=pstar[0:2])
                    J_pos_std = 0.
                else:
                    J_pos, J_pos_std = range_J(Jp, p=p, pstd=dp_std, pstar=pstar[0:2])

                if abs(gamma[3]) < 1e-6:
                    J_vel = Jv(p=p, pdot=jet.v, pstar=pstar[0:2])
                    J_vel_std = 0.
                else:
                    J_vel, J_vel_std = range_J(Jv, p=p, pdot=jet.v, pstd=dp_std, pstar=pstar[0:2])

                if abs(gamma[5]) < 1e-6:
                    J_sample = 0.
                else:
                    J_sample = self.__J_sample__(self, pos, new_pos, pstar, target_alt)

                J_fuel = 0.
                # J_i = np.dot(np.array([J_pos, J_vel, J_pos_std**2, J_vel_std**2, J_fuel, J_sample]), gamma) + J
                J_i = J_pos * gamma[0] + J
                if depth == 1:
                    J_i += J_vel * gamma[1]
                if J_i > self.lowest_J_yet:
                    return
                policy_i = np.append(np.array(policy), np.array(target_alt))
                self.__tree_search__(new_pos, this_key, u, T, pstar, depth-1, J_i, policy_i, gamma)

    def __find_altitudes_for_sampling__(self, N, threshold):
        x_test = np.zeros(N)
        y_test = np.zeros(N)
        z_test = np.linspace(self.lower, self.upper, N)
        p_test = np.array([x_test, y_test, z_test])
        vx, vy, std_x, std_y = self.predict(p_test)

        x_local_max = self.__local_max_idx__(std_x)
        y_local_max = self.__local_max_idx__(std_y)
        x_big_enough = np.array(std_x > threshold * abs(vx.reshape(N)), dtype=np.int)
        y_big_enough = np.array(std_y > threshold * abs(vy.reshape(N)), dtype=np.int)
        x_the_short_list = (x_local_max + x_big_enough) == 2
        y_the_short_list = (y_local_max + y_big_enough) == 2
        the_short_list = (x_the_short_list + y_the_short_list) > 0
        self.alts_to_sample = z_test[the_short_list]

    def __local_max_idx__(self, x):
        x_compare_to_prev = np.append(0, (x[1:] - x[:-1]))
        x_compare_to_next = np.append((x[:-1] - x[1:]), 0)
        x_bigger_than_prev = np.array(x_compare_to_prev > 0, dtype=np.int)
        x_bigger_than_next = np.array(x_compare_to_next > 0, dtype=np.int)
        x_local_max = (x_bigger_than_next + x_bigger_than_prev) == 2
        return np.array(x_local_max, dtype=np.int)

    def __delta_p_between_jetstreams__(self, u):
        if self.off_nominal:
            jets = self.jets_expectation
            jet_vals = self.jets_expectation.jetstreams.values()
            jet_keys = self.jets_expectation.jetstreams.keys()
        else:
            jets = self.jets
            jet_vals = self.jets.jetstreams.values()
            jet_keys = self.jets.jetstreams.keys()
        delta_p = np.zeros([len(jet_vals),len(jet_vals),2])
        delta_std = np.zeros([len(jet_vals),len(jet_vals),2])
        N = 500
        x_test = np.zeros(N)
        y_test = np.zeros(N)
        for i, key in enumerate(jet_keys):
            jets.jetstreams[key].set_id(i)
        for i, jet1 in enumerate(jet_vals):
            for j, jet2 in enumerate(jet_vals):
                if i == j:
                    z_test = jet1.avg_alt
                    p_test = np.array([0, 0, z_test])
                    vx, vy, std_x, std_y = self.predict(p_test)
                    d_std = np.array([std_x, std_y]).reshape(2)
                    delta_p[jet1.id,jet2.id] = np.zeros(2)
                    delta_p[jet2.id,jet1.id] = np.zeros(2)
                    delta_std[jet1.id,jet2.id] = d_std
                    delta_std[jet2.id,jet1.id] = d_std
                if i < j:
                    z1 = jet1.avg_alt
                    z2 = jet2.avg_alt
                    z_test = np.linspace(z1, z2, N)
                    p_test = np.array([x_test, y_test, z_test])
                    # vx, vy = self.ev(p_test)
                    vx, vy, std_x, std_y = self.predict(p_test)
                    mean_vx = np.mean(vx)
                    mean_vy = np.mean(vy)
                    dz = abs(z2 - z1)
                    t = abs(dz / u)
                    scaling_factor = abs((z_test[1] - z_test[0]) / u)
                    mean_std_x = np.mean(std_x**2)
                    mean_std_y = np.mean(std_y**2)
                    dx_std = np.sqrt(mean_std_x * t)
                    dy_std = np.sqrt(mean_std_y * t)
                    dx = mean_vx * t
                    dy = mean_vy * t
                    dp = np.array([dx, dy])
                    dp_std = np.array([dx_std, dy_std]).reshape(2)
                    delta_p[jet1.id,jet2.id] = dp
                    delta_p[jet2.id,jet1.id] = dp
                    delta_std[jet1.id,jet2.id] = dp_std
                    delta_std[jet2.id,jet1.id] = dp_std
                    # print(d_std)
        if self.off_nominal:
            self.delta_p_expectation = delta_p
            self.delta_std_expectation = delta_std
        else:
            self.delta_p = delta_p
            self.delta_std = delta_std

    def __min_leaf_J__(self, *args, **kwargs):
        min_J = np.inf
        for leaf_val in self.leaves.values():
            if leaf_val[2] < min_J:
                min_J = leaf_val[2]
        return min_J

    def __min_leaf__(self, *args, **kwargs):
        min_J = self.__min_leaf_J__()
        for leaf_key in self.leaves.keys():
            this_J = self.leaves[leaf_key][2]
            if abs(this_J - min_J) < 1e-6:
                return leaf_key
        return False

    def __best_pol__(self, *args, **kwargs):
        p = parsekw(kwargs, 'p', None)
        min_leaf = self.__min_leaf__()
        best_pol, best_J = self.__pol__(leaf=min_leaf)
        if self.off_nominal:
            best_pol = self.__sample_destination__(p, self.jets_expectation.find(best_pol[1]))
            best_J = 0.
        return best_pol, best_J

    def __pol__(self, *args, **kwargs):
        leaf = parsekw(kwargs, 'leaf', None)
        return_p = parsekw(kwargs, 'return_p', False)
        pos = self.backedges[leaf][1]
        cost = self.backedges[leaf][2]
        pol = np.array(pos)
        J = np.array(cost)
        stem = self.backedges[leaf][0]
        p0 = np.array(self.backedges[leaf][3])
        p1 = np.array(self.backedges[leaf][4])
        while True:
            pos = self.backedges[stem][1]
            cost = self.backedges[stem][2]
            pol = np.append(pos, pol)
            J = np.append(cost, J)
            p0 = np.append(self.backedges[stem][3], p0)
            p1 = np.append(self.backedges[stem][4], p1)
            if self.backedges[stem][0] == stem:
                break
            stem = self.backedges[stem][0]
        if return_p:
            return pol, J, p0, p1
        return pol, J

    def plot(self, *args, **kwargs):
        ax = parsekw(kwargs, 'ax', -1)
        plt.sca(ax)
        plt.cla()
        n_leaves = len(self.leaves.keys())
        gray = 0.8
        dgray = (gray - 0.5) / n_leaves
        # min_J = self.__min_leaf_J__()
        for leaf in self.leaves.keys():
            p, J = self.__pol__(leaf=leaf)
            # ax.semilogx(J, p*1e-3, c=np.ones(3)*gray)
            # ax.semilogx(J[-1], p[-1]*1e-3, 'o', c=np.ones(3)*gray)
            ax.plot(J, p*1e-3, c=np.ones(3)*gray)
            ax.plot(J[-1], p[-1]*1e-3, 'o', c=np.ones(3)*gray)
            gray -= dgray
            # if abs(self.leaves[leaf][2] - min_J) < 1e-6:
                # chosen_J = J
                # chosen_p = p
                # min_J = J[0]
        best_leaf = self.__min_leaf__()
        chosen_p, chosen_J = self.__pol__(leaf=best_leaf)
        # ax.semilogx(chosen_J, chosen_p*1e-3, '-k', linewidth=4)
        # ax.semilogx(chosen_J[-1], chosen_p[-1]*1e-3, 'ok')
        ax.plot(chosen_J, chosen_p*1e-3, '-k', linewidth=4)
        ax.plot(chosen_J[-1], chosen_p[-1]*1e-3, 'ok')
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlim([0, ax.get_xlim()[1]])
        # ax.set_xlim([3*1e18, 4*1e18])

    def plot_paths(self, *args, **kwargs):
        ax = parsekw(kwargs, 'ax', -1)
        p = parsekw(kwargs, 'p', None)
        h = []
        best_leaf = self.__min_leaf__()
        gray = 0.8
        n_leaves = len(self.leaves.keys())
        dgray = (gray - 0.5) / n_leaves
        for leaf_key in self.leaves.keys():
            pol, J, p0, p1 = self.__pol__(leaf=leaf_key, return_p=True)
            if leaf_key == best_leaf:
                best_p0 = p0
                best_p1 = p1
                best_gray = gray
            else:
                handle, = ax.plot(np.array(p1) * 1e-3, np.array(p0) * 1e-3, c=np.ones(3)*gray, linewidth=0.5)
                h.append(handle)
                handle, = ax.plot(np.array(p1)[-1] * 1e-3, np.array(p0)[-1] * 1e-3, 'o', c=np.ones(3)*gray, markersize=2)
                h.append(handle)
            gray -= dgray
        handle, = ax.plot(np.array(best_p1) * 1e-3, np.array(best_p0) * 1e-3, c='k', linewidth=0.5)
        h.append(handle)
        handle, = ax.plot(np.array(best_p1)[-1] * 1e-3, np.array(best_p0)[-1] * 1e-3, 'o', c='k', markersize=2)
        h.append(handle)
        return h

""" HASN'T BEEN USED IN A WHILE """
class LocalDynamicProgrammingPlanner(WindAwarePlanner):
    def __init__(self, *args, **kwargs):
        pstar = parsekw(kwargs, 'pstar', np.zeros(3))
        dp = parsekw(kwargs, 'dp', np.zeros(3))
        dt = parsekw(kwargs, 'dt', 180.0)
        dims = parsekw(kwargs, 'dims', np.ones(3))
        gamma = parsekw(kwargs, 'gamma', 0.9)
        u = parsekw(kwargs, 'u', 5.0)
        WindAwarePlanner.__init__(  self,
                                    field=kwargs.get('field'),
                                    lower=kwargs.get('lower'),
                                    upper=kwargs.get('upper'),
                                    streamsize=kwargs.get('streamsize'))
        np.set_printoptions(threshold='nan')
        self.dp = dp
        self.pstar = pstar
        self.__setup__(pstar, dp, dims)
        self.__delta_p_between_jetstreams__(u)
        self.__T_all__(u, dt)
        self.__R_all__(pstar)
        self.__policy_iteration__(u, pstar, gamma, dt)

    def plan(self, loon):
        state, i = self.__get_nearest_state__(loon.get_pos())
        return self.policy[i]

    def __setup__(self, pstar, dp, dims):
        """
        parameter dims 3-element vector, number of discrete elements in each direction
        """
        # Setup discretized action space (i.e. jetstreams)
        self.__setup_action_space__()
        # Setup discretized state space
        dims[2] = len(self.action_space)
        self.__setup_state_space__(pstar, dp, dims)
        # Setup policy container (3D matrix)
        self.policy = self.action_space[0] * np.ones(len(self.state_space))
        pass

    def __setup_state_space__(self, pstar, dp, dims):
        x = np.linspace(pstar[0] - dp[0], pstar[0] + dp[0], dims[0])
        y = np.linspace(pstar[1] - dp[1], pstar[1] + dp[1], dims[1])
        # z = np.linspace(pstar[2] - dp[2], pstar[2] + dp[2], dims[2])
        z = self.action_space
        states = np.meshgrid(x, y, z)
        n = np.prod(dims)
        m = len(dims)
        print(n)
        self.state_space = np.reshape(states, [m, n]).T
        self.state_space = np.append(self.state_space, [np.inf * np.ones(3)], axis=0)

    def __setup_action_space__(self):
        jetstreams = self.jets.jetstreams.values()
        n = len(jetstreams)
        actions = np.zeros(n)
        for i, jetstream in enumerate(jetstreams):
            actions[i] = jetstream.avg_alt
        self.action_space = actions

    def __policy_iteration__(self, u, pstar, gamma, dt):
        prev_policy = copy.deepcopy(self.policy) + 1
        while not (prev_policy == self.policy).all():
            print(np.floor(self.policy.T))
            prev_policy = copy.deepcopy(self.policy)
            print("T_pi...")
            T_pi = self.__T_pi__(u, dt)
            print("R_pi...")
            R_pi = self.__R_pi__(pstar)
            print("V_pi...")
            V_pi = self.__V_pi__(T_pi, R_pi, gamma)
            for i, state in enumerate(self.state_space):
                if i % 100 == 0:
                    print("\t" + str(i) + "/" + str(len(self.state_space)))
                best_action = state[2]
                best_V = -np.inf
                for j, action in enumerate(self.action_space):
                    T = self.T[i,j]
                    R = self.R[i,j]
                    V = R + gamma * T * V_pi
                    if V > best_V:
                        best_V = V
                        best_action = action
                self.policy[i] = best_action

    def __V_pi__(self, T, R, gamma):
        I = np.eye(len(self.state_space))
        M = (np.matrix(I) - gamma * np.matrix(T))**(-1)
        V = M * np.matrix(R).T
        return V

    def __R_pi__(self, pstar):
        R = np.zeros(len(self.state_space))
        for i, state in enumerate(self.state_space):
            action = self.policy[i]
            for j, a in enumerate(self.action_space):
                if (a - action) < 1e-3:
                    R[i] = self.R[i,j]
                    break
        return R

    def __T_pi__(self, u, dt):
        T = np.zeros([len(self.state_space), len(self.state_space)])
        for i, state in enumerate(self.state_space):
            action = self.policy[i]
            for j, a in enumerate(self.action_space):
                if (a - action) < 1e-3:
                    T[i] = self.T[i,j]
                    break
        return T

    def __R__(self, s, a, pstar):
        if (s == np.inf).any():
            return -1e12
        jetstream = self.jets.find(s[2])
        J_pos = np.sqrt(np.sum((pstar[0:2] - s[0:2])**2))*1e-2
        J_vel = WindAwarePlanner.__cost_of_jetstream__(self, s, pstar, jetstream)*1e1
        J_fuel = (a - s[2])**2*1e-6
        J = J_pos + J_vel + J_fuel
        R = -J
        return R

    def __T__(self, s, a, u, dt):
        T = np.zeros(len(self.state_space))
        if (s == np.inf).any():
            T[-1] = 1
            return T
        jetstream_s = self.jets.find(s[2])
        jetstream_a = self.jets.find(a)
        alt_s = jetstream_s.avg_alt
        alt_a = jetstream_a.avg_alt
        if abs(alt_s - alt_a) < 1e-3:
            pos = s
            new_state, idx = self.__get_nearest_state__(pos)
            out_of_range = False
            while (new_state == s).all() and not out_of_range:
                pos += jetstream_s.ride_for_dt(dt)
                new_state, idx = self.__get_nearest_state__(pos)
                out_of_range = (np.array(idx) < 0).any()
        else:
            dp = self.delta_p[jetstream_s.id, jetstream_a.id]
            dz = alt_a - alt_s
            new_state = s + np.append(dp, dz)
            new_state, idx = self.__get_nearest_state__(new_state)
        T[idx] = 1
        return T

    def __T_all__(self, u, dt):
        n = len(self.state_space)
        m = len(self.action_space)
        self.T = np.zeros([n, m, n])
        print("T_all...")
        for i, state in enumerate(self.state_space):
            if i % 100 == 0:
                print("\t" + str(i) + "/" + str(len(self.state_space)))
            for j, action in enumerate(self.action_space):
                self.T[i,j] = self.__T__(state, action, u, dt)

    def __R_all__(self, pstar):
        n = len(self.state_space)
        m = len(self.action_space)
        self.R = np.zeros([n, m])
        print("R_all...")
        for i, state in enumerate(self.state_space):
            if i % 100 == 0:
                print("\t" + str(i) + "/" + str(len(self.state_space)))
            for j, action in enumerate(self.action_space):
                self.R[i,j] = self.__R__(state, action, pstar)

    def __get_nearest_state__(self, s):
        out_of_range = np.linalg.norm(self.dp) < np.linalg.norm(self.pstar - s)
        if out_of_range:
            return self.state_space[-1], -1
        min_d = np.inf
        nearest_state = np.zeros(len(s))
        nearest_idx = 0
        d = np.sum((s - self.state_space)**2, axis=1)
        min_d = np.min(d)
        idx = (d == min_d)
        state = self.state_space[idx]
        return state, idx

    def plot(self, u, dt):
        policy_plot = plt.figure().gca(projection='3d')
        for i, state in enumerate(self.state_space):
            if i % 100 == 0:
                print("\t" + str(i) + "/" + str(len(self.state_space)))
            T = np.array(self.__T__(state, self.policy[i], u, dt), dtype=bool)
            next_state = np.squeeze(np.array(self.state_space[T]))
            # if (state == next_state).all() or (next_state == np.inf).any():
                # print("WHAT")
            x = np.array([state[0], next_state[0]])
            y = np.array([state[1], next_state[1]])
            z = np.array([state[2], next_state[2]])
            policy_plot.plot(x, y, z)
        plt.show()

    def __delta_p_between_jetstreams__(self, u):
        jets = self.jets.jetstreams.values()
        self.delta_p = np.zeros([len(jets),len(jets),2])
        N = 500
        x_test = np.zeros(N)
        y_test = np.zeros(N)
        for i, jet1 in enumerate(self.jets.jetstreams.values()):
            jet1.set_id(i)
            for j, jet2 in enumerate(self.jets.jetstreams.values()):
                if i < j:
                    z1 = jet1.avg_alt
                    z2 = jet2.avg_alt
                    z_test = np.linspace(z1, z2, N)
                    p_test = np.array([x_test, y_test, z_test])
                    vx, vy = self.ev(p_test)
                    mean_vx = np.mean(vx)
                    mean_vy = np.mean(vy)
                    dz = abs(z2 - z1)
                    t = dz / u
                    dx = mean_vx * t
                    dy = mean_vy * t
                    dp = np.array([dx, dy])
                    self.delta_p[i,j] = dp
                    self.delta_p[j,i] = dp
