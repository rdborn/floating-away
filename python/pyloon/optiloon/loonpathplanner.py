import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import copy
from copy import deepcopy

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyloon.multiinputloon import MultiInputLoon as Loon
from pyutils.pyutils import parsekw, hash3d, hash4d, rng, downsize
from pyutils import pyutils
from pyflow.pystreams import VarThresholdIdentifier as JSI

class LoonPathPlanner:
    def __init__(self, *args, **kwargs):
        """
        Initialize a general loon path planner object.

        kwarg 'field' is the pyflow.flowfields.FlowField used in this simulation.
        kwarg 'lower' is the lowest allowable altitude.
        kwarg 'upper' is the highest allowable altitude.
        """

        field = parsekw(kwargs, 'field', 0.0) # wind field object
        self.lower = parsekw(kwargs, 'lower', 0.0)
        self.upper = parsekw(kwargs, 'upper', 100)
        self.retrain(field=field)

    def retrain(self, *args, **kwargs):
        """
        Retrain the path planner using a new flow field.

        kwarg 'field' is the pyflow.flowfields.FlowField used in this simulation.
        """

        field = parsekw(kwargs, 'field', 0.0) # wind field object

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
        """
        Estimate the predicted means and standard deviations of the flow field at the provided point.

        parameter p 3D point at which to make the prediction.
        return mean of flow in x direction
        return mean of flow in y direction
        return standard deviation of flow in x direction
        return standard deviation of flow in y direction
        """

        mean_fx, std_fx = self.GPx.predict(np.atleast_2d(np.array(p)[self.mask]), return_std=True)
        mean_fy, std_fy = self.GPy.predict(np.atleast_2d(np.array(p)[self.mask]), return_std=True)
        return mean_fx, mean_fy, std_fx, std_fy

    def ev(self, p):
        """
        Estimate only the expected value of the flow field at the provided point.

        parameter p 3D point at which to make the prediction.
        return mean of flow in x direction
        return mean of flow in y direction
        """
        p_pred = np.atleast_2d(np.array(p)[self.mask])
        p_pred = p_pred.T if np.shape(p_pred)[0] == 1 else p_pred
        mean_fx = self.GPx.predict(p_pred, return_std=False)
        mean_fy = self.GPy.predict(p_pred, return_std=False)
        return mean_fx, mean_fy

    def __cost__(self, p, pstar):
        """
        Evaluate the cost function at the provided point for the provided goal.

        parameter p 3D point at which to evaluate cost function.
        parameter pstar 3D point representing the goal location.
        return cost function evaluated at p for goal pstar.
        """

        return np.linalg.norm(np.subtract(p[0:2],pstar[0:2]))

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

    def __drag_force__(self, loon, v):
        """
        NOT SUPPORTED
        Calculate drag force for a given balloon and wind velocity.

        parameter loon balloon object for which to calculate drag force.
        parameter v wind speed for which to calculate drag force.
        return drag force for loon in wind of speed v.
        """
        # TODO: change to wind-relative velocity
        rho = 0.25 # air density
        return v * abs(v) * rho * loon.A * loon.Cd / 2

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

    def plan(self, loon, pstar, depth):
        """
        Plan the path for the balloon.

        parameter loon balloon object for which to plan.
        parameter pstar 3D point, goal location
        parameter depth recursion depth for Monte Carlo tree search
        return optimal policy (discrete time control effort in reverse order)
        """

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

class PlantInvertingController(LoonPathPlanner):
    def plan(self, *args, **kwargs):
        """
        Calculate control effort to accelerate towards the origin.

        kwarg loon balloon object for which to plan
        return plant-inverting control effort (i.e. attempt to accelerate towards origin)
        """

        loon = parsekw(kwargs, 'loon', None)
        p = np.array(loon.get_pos())
        if loon == None:
            warning("Must specify loon.")
        J_x = self.__differentiate_gp__(self.GPx, p[2])
        J_y = self.__differentiate_gp__(self.GPy, p[2])
        J = np.array([J_x, J_y])
        x = p[0:2]
        num = np.inner(J, -x / np.inner(x, x))
        den = np.inner(J,J)
        u = 10000.0 * num / den
        print("u: " + str(u))
        u = pyutils.saturate(u, 5)
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
        M = np.dot(np.linalg.inv(K1) + np.linalg.inv(K2) + np.eye(len(x))*sigma, y)
        return K1, K2

    def __differentiate_gp__(self, GP, xstar):
        """
        Differentiate a Gaussian Process.

        parameter GP Gaussian Process to differentiate.
        parameter xstar test point at which to differentiate.
        return derivative of GP at xstar
        """
        # NOTE: Cannot distinguish between different kernel types
        #       (i.e. function is hardcoded for two summed RBF kernels)

        K1, K2 = self.__recreate_gp__(GP)
        y = GP.y_train_
        theta = np.exp(GP.kernel_.theta)
        M1 = np.dot(np.linalg.inv(K1),y)
        M2 = np.dot(np.linalg.inv(K2),y)
        Kstar1, Kstar2 = self.__kstar_gp__(GP, xstar)
        # print(np.dot(Kstar1.T,M1)+np.dot(Kstar2.T,M2))
        # print(GP.predict(xstar))
        L1 = theta[1]
        L2 = theta[3]
        x = GP.X_train_
        dkdx =  np.dot(-((xstar - x) / L1**2).T, Kstar1*M1) + \
                np.dot(-((xstar - x) / L2**2).T, Kstar2*M2)
        return np.squeeze(dkdx)

class WindAwarePlanner(LoonPathPlanner):
    def __init__(self, *args, **kwargs):
        """
        Initialize a wind-aware path planner for a balloon.

        kwarg 'field' is the pyflow.flowfields.FlowField used in this simulation.
        kwarg 'lower' is the lowest allowable altitude.
        kwarg 'upper' is the highest allowable altitude.
        kwarg 'streamres' number of points to sample from air column.
        kwarg 'streammin' minimum altitude from which to sample.
        kwarg 'streammax' maximum altitude from which to sample.
        kwarg 'threshold' variance threshold for identifying unique jetstreams.
        kwarg 'streamsize' minimum number of sample points for a jetstream to be classified as such.
        """

        LoonPathPlanner.__init__(   self,
                                    field=kwargs.get('field'),
                                    lower=kwargs.get('lower'),
                                    upper=kwargs.get('upper'))
        streamres = parsekw(kwargs, 'streamres', 500)
        streammax = parsekw(kwargs, 'streammax', 0)
        streammin = parsekw(kwargs, 'streammin', 30000)
        threshold = parsekw(kwargs, 'threshold', 0.01)
        streamsize = parsekw(kwargs, 'streamsize', 20)

        alt = np.linspace(streammin, streammax, streamres)
        streamdir = np.zeros(streamres)
        streammag = np.zeros(streamres)
        for i, z in enumerate(alt):
        	vx, vy = self.ev(np.array([0, 0, z]))
        	magnitude = np.sqrt(vx**2 + vy**2)
        	direction = np.arctan2(vy, vx)
        	streammag[i] = magnitude
        	streamdir[i] = np.cos(direction)
        data = np.array([streammag, streamdir, alt]).T
        self.jets = JSI(data=data, threshold=threshold, streamsize=streamsize)

    def plan(self, loon, pstar):
        """
        Calculate and return which jetstream for a balloon to travel to.

        parameter loon balloon for which to plan.
        parameter pstar 3D point, goal location.
        return altitude of center of most desirable jetstream.
        """

        # vals = np.array(self.jets.jetstreams.values())
        # z_test = vals[:,0]
        # theta = np.arccos(vals[:,2])
        # theta = self.__wind_dir__(z_test)
        phi = self.__desired_dir__(loon, pstar)
        # candidates = np.cos(phi - theta)
        # candidates = self.__smooth__(phi, theta)
        # target = self.__min_climb__(loon, z_test, candidates)
        target, J = self.__find_best_jetstream__(loon, pstar, 5)
        print("Target altitude: " + str(target.avg_alt))
        print("Direction at target: " + str((target.direction * 180.0 / np.pi) % 360))
        print("Desired direction: " + str((phi * 180.0 / np.pi - 180.0) % 360))
        return target.avg_alt

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

    def __wind_dir__(self, z_test):
        """
        Calculate wind direction at various altitudes.

        parameter z_test altitudes at which to calculate (estimated) wind direction.
        return estimated wind directions at altitudes in z_test
        """

        vx_test = self.GPx.predict(np.atleast_2d(np.array(z_test)).T, return_std=False)
        vy_test = self.GPy.predict(np.atleast_2d(np.array(z_test)).T, return_std=False)
        theta = np.arctan2(vy_test, vx_test)
        return theta

    def __find_best_jetstream__(self, loon, pstar, u):
        n_streams = len(self.jets.jetstreams)
        J_jetstreams = np.inf * np.ones(n_streams)
        J_fuel_jetstreams = np.inf * np.ones(n_streams)
        J = np.inf * np.ones(n_streams)
        pos = loon.get_pos()
        x_loon = pos[0]
        y_loon = pos[1]
        z_jets = np.zeros(n_streams)
        best_jet = self.jets.jetstreams.values()[0]
        best_J = np.inf
        for i, jet in enumerate(self.jets.jetstreams.values()):
            z_jets[i] = jet.avg_alt
            dp, dz = self.__cost_to_altitude__(loon, z_jets[i], u)
            dx = dp[0]
            dy = dp[1]
            new_pos = np.array([dx, dy, dz]) + pos
            J_fuel_jetstreams[i] = dz*1e-3
            J_jetstreams[i] = self.__cost_of_jetstream__(new_pos, pstar, jet)
            J[i] = J_fuel_jetstreams[i] + J_jetstreams[i]
            if J[i] < best_J:
                best_J = J[i]
                best_jet = jet
            print("Alt: " + str(jet.avg_alt) + "\t\tJf: " + str(np.int(J_fuel_jetstreams[i])) + "\t\tJj: " + str(np.int(J_jetstreams[i])))
        return best_jet, best_J

    def __phiddot__(self, p, phat, pdot):
        phidot = np.dot(phat, pdot) * phat
        phiddot = (((np.linalg.norm(pdot)**2 - 2 * np.linalg.norm(phidot)**2)) * phat + np.linalg.norm(phidot) * pdot) / np.linalg.norm(p)
        return phiddot

    # def __costs_of_each_jetstream__(self, loon, pstar):
    #     pos = np.array(loon.get_pos())
    #     p = pos[0:2]
    #     z_loon = pos[2]
    #     phat = p / np.linalg.norm(p)
    #     vals = np.array(self.jets.jetstreams.values())
    #     z = vals[:,0]
    #     magnitude = vals[:,1]
    #     direction = vals[:,2]
    #     vx = magnitude * np.cos(direction)
    #     vy = magnitude * np.sin(direction)
    #     pdots = np.squeeze(np.array([vx, vy]).T)
    #     pstar = pstar[0:2]
    #     phi = p - pstar
    #     J = np.inf * np.ones(len(pdots))
    #     for i, pdot in enumerate(pdots):
    #         phiddot = self.__phiddot__(p, phat, pdot)
    #         J[i] = np.dot(p, p) + np.dot(p, pdot) + np.dot(phiddot, phiddot) # + (z_loon - z[i])**2
    #     return J, z

    def __cost_of_jetstream__(self, pos, pstar, jet):
        # Get the balloon's current position and store its lateral and vertical
        # positions separately
        # pos = np.array(loon.get_pos())
        p = pos[0:2]
        z_loon = pos[2]
        # Get the jetstream in question and store its magnitude and direction
        # jet = self.jets.find(z_jet)
        magnitude = jet.magnitude
        direction = jet.direction
        # Calculate the components of the wind velocity and store it as the
        # time derivative of the balloon's position
        vx = magnitude * np.cos(direction)
        vy = magnitude * np.sin(direction)
        pdot = np.squeeze(np.array([vx, vy]).T)
        pdothat = pdot / np.linalg.norm(pdot)
        # Calculate the unit vector along the balloon's lateral position vector
        phat = p / np.linalg.norm(p)
        # Get the lateral position of the goal point
        pstar = pstar[0:2]
        # Translate the coordinate system such that pstar is at the origin
        phi = p - pstar
        phihat = phi / np.linalg.norm(phi)
        # Calculate the acceleration of the balloon towards the origin
        phiddot = self.__phiddot__(p, phat, pdot)
        # Calculate the cost of this jetstream at this position with this goal
        # position as...
        # + the magnitude of the distance from the goal
        #       (encouragement to be near the goal)
        # + velocity away from the goal
        #       (encouragement to move towards the goal)
        # + the magnitude of the acceleration towards the goal
        #       (encouragement to move in a controlled manner with respect to the goal)
        # print("phi: " + str(phi) + "\t\tpdot: " + str(pdot))
        J_position = np.sqrt(np.dot(phi, phi))*1e-3*0
        J_velocity = (np.dot(phihat, pdothat)+1)*np.linalg.norm(p)*1e-1
        J_accel = np.dot(phiddot, phiddot)*0
        J = J_position + J_velocity + J_accel
        # Return the calculated cost
        return J

    # def __dp_btwn_jetstreams__(self, loon, pstar, u):
    #     vals = np.array(self.jets.jetstreams.values())
    #     altitude = vals[:,0]
    #     dp_next = np.zeros([len(altitude),2])
    #     dp = np.zeros([len(altitude), len(altitude), 2])
    #     for i, z in enumerate(altitude):
    #         if (len(altitude) - i) > 1:
    #             z_test = np.linspace(z[i], z[i+1], 100)
    #             vx, vy = self.ev(z_test)
    #             mean_vx = np.mean(vx)
    #             mean_vy = np.mean(vy)
    #             dz = z[i+1] - z[i]
    #             t = dz / u
    #             dx = mean_vx * t
    #             dy = mean_vy * t
    #             dp_next[i] = np.array([dx, dy])
    #     for i in range(len(dp_next)):
    #         for j in range(i, len(dp_next)):
    #             dp_ij = 0.0
    #             for k in range(i, j):
    #                 dp_ij += dp_next[k]
    #             dp[i,j] = dp_ij
    #             dp[j,i] = dp_ij
    #     self.dp_jetstreams = dp
    #     return dp

    def __cost_to_altitude__(self, loon, z, u):
        pos = loon.get_pos()
        z_loon = pos[2]
        N = 500
        x_test = np.zeros(N)
        y_test = np.zeros(N)
        z_test = np.linspace(z_loon, z, N)
        p_test = np.array([x_test, y_test, z_test])
        vx, vy = self.ev(p_test)
        mean_vx = np.mean(vx)
        mean_vy = np.mean(vy)
        dz = abs(z - z_loon)
        t = dz / u
        dx = mean_vx * t
        dy = mean_vy * t
        dp = np.array([dx, dy])
        return dp, dz
