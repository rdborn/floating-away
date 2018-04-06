import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import copy

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng, downsize
from pyutils import pyutils
from pyflow.pystreams import VarThresholdIdentifier as JSI
from optiloon.fieldestimator import GPFE, KNN1DGP, Multi1DGP

class NaivePlanner:
    def __init__(self, *args, **kwargs):
        self.last_sounding_position = np.inf * np.ones(2)
        self.last_sounding_time = -np.inf
        self.threshold = parsekw(kwargs, 'resamplethreshold', 30000)
        self.trusttime = parsekw(kwargs, 'trusttime', 3)
        self.sampled_points = []
        self.tlast = 0.0
        self.fieldestimator = 'naive'

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

    def plan(self, *args, **kwargs):
        pstar = parsekw(kwargs, 'pstar', np.zeros(2))
        loon = parsekw(kwargs, 'loon', None)
        tcurr = parsekw(kwargs, 'tcurr', -1)
        gamma = parsekw(kwargs, 'gamma', 1e7)
        curr_pos = np.array(loon.get_pos()[0:2])
        last_pos = self.last_sounding_position
        out_of_range = (np.linalg.norm(curr_pos - last_pos) > self.threshold)
        been_too_long = ((tcurr - self.tlast) / 3600) > self.trusttime
        need_to_resample = out_of_range or been_too_long
        if need_to_resample:
            self.tlast = tcurr
            self.last_sounding_position = curr_pos
            return np.array([-1])
        min_J = np.inf
        best_alt = 0
        for i, d in enumerate(self.data):
            alt = d[0]
            v = d[1:]
            J_pos = np.sqrt((np.sum((curr_pos[0:2] - pstar[0:2])**2)))
            J_vel = self.__cost_of_vel__(curr_pos, pstar, v)
            J = np.log(J_pos * (J_vel * gamma + 1) + 1)
            if J < min_J:
                min_J = J
                best_alt = alt
        return np.array([best_alt])

    def __cost_of_vel__(self, pos, pstar, vel):
        """
        Calculate the terminal cost of the given jetstream at the given position
        for the given set point.

        parameter pos position at which to calculate cost of jetstream
        parameter pstar set point/goal position
        parameter jet jetstream for which to calculate cost
        return terminal cost of jet at pos for pstar
        """

        # SETTING UP/INITIALIZATION
        # Get the balloon's current position and store its lateral and vertical
        # positions separately
        p = pos
        # Calculate the components of the wind velocity and store it as the
        # time derivative of the balloon's position
        vx = vel[0]
        vy = vel[1]
        pdot = np.squeeze(np.array([vx, vy]).T)
        pdothat = pdot / np.linalg.norm(pdot)
        # Calculate the unit vector along the balloon's lateral position vector
        norm_p = np.linalg.norm(p)
        # phat = p / norm_p if norm_p > 0 else p
        # Get the lateral position of the goal point
        pstar = pstar[0:2]
        # Translate the coordinate system such that pstar is at the origin
        phi = p - pstar
        norm_phi = np.linalg.norm(phi)
        phihat = phi / norm_phi if norm_phi > 0 else phi
        # CALCULATE COST:
        J_velocity = (np.dot(phihat, pdothat)+1)

        # Return the calculated cost
        return J_velocity

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
        vx, vy, coords = self.__parse_field__(field)

        print(coords.shape)
        print("\tTraining GPx")
        self.vx_estimator.fit(X=coords,
                            y=vx,
                            kernel=self.kernel,
                            n_restarts_optimizer=self.n_restarts_optimizer)
        print("\tTraining GPy")
        self.vy_estimator.fit(X=coords,
                            y=vy,
                            kernel=self.kernel,
                            n_restarts_optimizer=self.n_restarts_optimizer)

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

    """ NOT SUPPORTED """
    def __cost__(self, p, pstar):
        """
        Evaluate the cost function at the provided point for the provided goal.

        parameter p 3D point at which to evaluate cost function.
        parameter pstar 3D point representing the goal location.
        return cost function evaluated at p for goal pstar.
        """

        return np.linalg.norm(np.subtract(p[0:2],pstar[0:2]))

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

    """ NOT SUPPORTED """
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
        J_x = self.__differentiate_gp__(self.vx_estimator.estimators[0], p[2])
        J_y = self.__differentiate_gp__(self.vy_estimator.estimators[0], p[2])
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
        streamdir = np.zeros(self.streamres)
        streammag = np.zeros(self.streamres)
        unbounded_angle = np.zeros(self.streamres)
        for i, z in enumerate(alt):
            vx, vy, std_x, std_y = self.predict(np.array([p[0], p[1], z]))
            magnitude = np.sqrt(vx**2 + vy**2)
            direction = np.arctan2(vy, vx)
            streammag[i] = magnitude
            streamdir[i] = direction
            vxmin, vxmax = pyutils.get_samesign_bounds(vx, std_x)
            vymin, vymax = pyutils.get_samesign_bounds(vy, std_y)
            unbounded_angle[i] = ((abs(vxmin) < 1e-6) or (abs(vxmax) < 1e-6)) and \
                                ((abs(vymin) < 1e-6) or (abs(vymax) < 1e-6))
        data = np.array([streammag, streamdir, alt, unbounded_angle]).T
        self.jets = JSI(data=data, threshold=self.threshold, streamsize=self.streamsize)

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
            J_jetstreams[i] = self.__cost_of_jetstream__(new_pos, pstar, jet)
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

    def __phiddot__(self, p, phat, pdot):
        """
        Calculate the acceleration towards the origin

        parameter p position at which to perform calculation
        parameter phat unit vector along p
        parameter pdot velocity at which to perform calculation
        return acceleration toward the origin
        """

        # TODO: Extend this to be acceleration towards a set point

        p = np.array(p)
        phat = np.array(phat)
        pdot = np.array(pdot)
        # Find velocity towards origin
        phidot = np.dot(phat, pdot) * phat
        # Find acceleration towards origin
        norm_p = np.linalg.norm(p)
        phiddot = (((np.linalg.norm(pdot)**2 - 2 * np.linalg.norm(phidot)**2)) * phat + np.linalg.norm(phidot) * pdot)
        phiddot = phiddot / norm_p if norm_p > 0 else phiddot
        # Return acceleration towards origin
        return phiddot

    def __cost_of_jetstream__(self, pos, pstar, jet):
        """
        Calculate the terminal cost of the given jetstream at the given position
        for the given set point.

        parameter pos position at which to calculate cost of jetstream
        parameter pstar set point/goal position
        parameter jet jetstream for which to calculate cost
        return terminal cost of jet at pos for pstar
        """

        # SETTING UP/INITIALIZATION
        # Get the balloon's current position and store its lateral and vertical
        # positions separately
        # pos = np.array(loon.get_pos())
        p = pos[0:2]
        z_loon = pos[2]
        pstar = pstar[0:2]
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
        norm_p = np.linalg.norm(p)
        # phat = p / norm_p if norm_p > 0 else p
        # Get the lateral position of the goal point
        pstar = pstar[0:2]
        # Translate the coordinate system such that pstar is at the origin
        phi = p - pstar
        norm_phi = np.linalg.norm(phi)
        phihat = phi / norm_phi if norm_phi > 0 else phi
        # Calculate the acceleration of the balloon towards the origin
        # phiddot = self.__phiddot__(p, phat, pdot)

        # CALCULATE COST:
        # Calculate the cost of this jetstream at this position with this goal
        # position as...
        # J_position = np.sqrt(np.dot(phi, phi))*1e-3*0
        J_velocity = (np.dot(phihat, pdothat)+1)
        # J_accel = np.dot(phiddot, phiddot)*0
        # J = J_position + J_velocity + J_accel
        J = J_velocity

        # Return the calculated cost
        return J

    def __accel_cost__(self, pos, pstar, jet):
        # CALCULATE COST:
        J_accel = np.linalg.norm(self.__accel__(pos, pstar, jet))
        J = J_accel
        # Return the calculated cost
        return J

    def __accel__(self, pos, pstar, jet):
        # SETTING UP/INITIALIZATION
        # Get the balloon's current position and store its lateral and vertical
        # positions separately
        # pos = np.array(loon.get_pos())
        p = pos[0:2]
        # Get the jetstream in question and store its magnitude and direction
        # jet = self.jets.find(z_jet)
        magnitude = jet.magnitude
        direction = jet.direction
        # Calculate the components of the wind velocity and store it as the
        # time derivative of the balloon's position
        vx = magnitude * np.cos(direction)
        vy = magnitude * np.sin(direction)
        pdot = np.squeeze(np.array([vx, vy]).T)
        # Calculate the unit vector along the balloon's lateral position vector
        norm_p = np.linalg.norm(p)
        phat = p / norm_p if norm_p > 0 else p
        # Calculate the acceleration of the balloon towards the origin
        phiddot = self.__phiddot__(p, phat, pdot)
        return phiddot

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
        self.__delta_p_between_jetstreams__(u)
        self.__find_altitudes_for_sampling__(100, 0.3)

    def __redo_jetstreams_etc__(self):
        WindAwarePlanner.__redo_jetstreams__(self, loon.get_pos())
        self.__delta_p_between_jetstreams__(u)
        self.__find_altitudes_for_sampling__(100, 0.3)

    def __incorporate_samples__(self, redo_stuff):
        if len(self.sampled_points) > 0:
            print("Omg new data!")
            new_X = np.array(self.sampled_points)[:,0]
            new_y_x = np.array(self.sampled_points)[:,1]
            new_y_y = np.array(self.sampled_points)[:,2]
            self.vx_estimator.add_data(X=new_X, y=new_y_x)
            self.vy_estimator.add_data(X=new_X, y=new_y_y)
            self.sampled_points = []
            if redo_stuff:
                self.__redo_jetstreams_etc__()

    def __reset_plan__(self):
        self.sequences = dict()
        self.backedges = dict()
        self.curr_key = 0
        self.leaves = dict()
        self.lowest_J_yet = np.inf
        self.nodes_expanded = 0

    def plan(self, *args, **kwargs):
        loon = parsekw(kwargs, 'loon', None)
        u = parsekw(kwargs, 'u', 0.0)
        T = parsekw(kwargs, 'T', 180.0)
        pstar = parsekw(kwargs, 'pstar', np.zeros(3))
        depth = parsekw(kwargs, 'depth', 0)
        gamma = parsekw(kwargs, 'gamma', 1e7)

        radius = 40000
        if self.vx_estimator.changing_estimators(p=loon.get_pos(), radius=radius):
            self.__incorporate_samples__(False)
            self.__redo_jetstreams_etc__()
        else:
            self.__incorporate_samples__(True)
        print(self.jets)
        self.__reset_plan__()
        pos = loon.get_pos()
        # self.__tree_search__(pos, 1, u, T, pstar, depth, 0.0, np.array([]), gamma)
        self.__tree_search_w_std__(pos, 1, u, T, pstar, depth, 0.0, np.array([]), gamma)
        best_leaf = self.__min_leaf__()
        best_pol, best_J = self.__pol__(leaf=best_leaf)
        # best_pol = self.sequences[min_J]
        print("\t\tNodes expanded: " + str(self.nodes_expanded))
        return best_pol

    def __get_stay_branch_length__(self, pos, pstar, jet):
        dp = np.zeros(2)
        prev_accel = WindAwarePlanner.__accel__(self, pos[0:2], pstar, jet)
        prev_accel_cost = WindAwarePlanner.__accel_cost__(self, pos[0:2], pstar, jet)
        new_accel_cost = WindAwarePlanner.__accel_cost__(self, pos[0:2]+dp, pstar, jet)
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
                new_accel_cost = WindAwarePlanner.__accel_cost__(self, pos[0:2]+dp, pstar, jet)
                d_accel_cost = abs(new_accel_cost - prev_accel_cost)
        else:
            total_T += T
            dp = np.array([vx, vy]) * total_T
        dp_std = self.delta_std[jet.id,j] * total_T
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
    def __hacky_way_to_avoid_bug_when_not_in_jetstream__(self, jet_id):
        buf = 10.0
        while jet_id < 0:
            buf += np.sign(buf) * 10.0
            buf *= -1
            this_jet = self.jets.find(pos[2] + buf)
            jet_id = this_jet.id
            print(buf)

    def __tree_search__(self, pos, prev_key, u, T, pstar, depth, J, policy, gamma):
        self.curr_key += 1
        this_key = self.curr_key
        back_key = prev_key
        val = np.array([back_key, pos[2], J])
        self.backedges[this_key] = val

        self.nodes_expanded += 1

        if depth == 0:
            self.leaves[this_key] = val
            if J < self.lowest_J_yet:
                self.lowest_J_yet = J
            if J in self.sequences.keys():
                print("AHH")
            self.sequences[J] = policy
            return
        this_jet = self.jets.find(pos[2])
        j = this_jet.id

        # HACK:
        self.__hacky_way_to_avoid_bug_when_not_in_jetstream__(j)

        jets = self.jets.jetstreams.values()
        for i, jet in enumerate(jets):
            if jet.avg_alt > self.lower and jet.avg_alt < self.upper:
                target_alt = jet.avg_alt
                if jet.id == j:
                    dp, dp_std = __get_stay_branch_length__(pos, pstar, jet)
                    J_fuel = 0.0
                else:
                    dp = self.delta_p[jet.id,j]
                    J_fuel = (target_alt - pos[2])**2*1e-6
                new_pos = np.append(pos[0:2] + dp, target_alt)
                J_pos = np.sqrt(np.sum((new_pos[0:2] - pstar[0:2])**2))
                J_vel = WindAwarePlanner.__cost_of_jetstream__(self, new_pos, pstar, jet)*gamma
                # J_i = J + J_pos + J_vel + J_fuel
                # J_i = np.log((J_pos * (J_vel + 1))) + J
                J_i = J_pos + J_vel + J
                if J_i > self.lowest_J_yet:
                    return
                policy_i = np.append(np.array(policy), np.array(target_alt))
                self.__tree_search__(new_pos, this_key, u, T, pstar, depth-1, J_i, policy_i, gamma)

    def __tree_search_w_std__(self, pos, prev_key, u, T, pstar, depth, J, policy, gamma):
        self.curr_key += 1
        this_key = self.curr_key
        back_key = prev_key
        val = np.array([back_key, pos[2], J])
        self.backedges[this_key] = val

        self.nodes_expanded += 1

        if depth == 0:
            self.leaves[this_key] = val
            if J < self.lowest_J_yet:
                self.lowest_J_yet = J
            if J in self.sequences.keys():
                print("AHH")
            self.sequences[J] = policy
            return
        this_jet = self.jets.find(pos[2])
        j = this_jet.id

        # HACK:
        self.__hacky_way_to_avoid_bug_when_not_in_jetstream__(j)

        jets = self.jets.jetstreams.values()
        for i, jet in enumerate(jets):
            if jet.avg_alt > self.lower and jet.avg_alt < self.upper:
                target_alt = jet.avg_alt
                if jet.id == j:
                    dp, dp_std = self.__get_stay_branch_length__(pos, pstar, jet)
                    J_fuel = 0.0
                else:
                    dp = self.delta_p[jet.id,j]
                    dp_std = self.delta_std[jet.id,j]
                    J_fuel = (target_alt - pos[2])**2
                new_pos = np.append(pos[0:2] + dp, target_alt)
                J_pos, J_pos_std = self.__range_J__(self.__J_pos__, pos=new_pos, std=dp_std, pstar=pstar, jet=jet)
                J_vel, J_vel_std = self.__range_J__(self.__J_vel__, pos=new_pos, std=dp_std, pstar=pstar, jet=jet)
                gamma = gamma * np.ones(6) if len(gamma) == 1 else gamma
                J_sample = 0.0 if gamma[-1] == 0 else self.__J_sample__(self, pos, new_pos, pstar, target_alt)
                # print("pos: " + str(J_pos) + "\tvel: " + str(J_vel) + "\tstd: " + str(J_pos_std))
                J_i = np.dot(np.array([J_pos, J_vel, J_pos_std**2, J_vel_std**2, J_fuel, J_sample]), gamma) + J
                if J_i > self.lowest_J_yet:
                    return
                policy_i = np.append(np.array(policy), np.array(target_alt))
                self.__tree_search_w_std__(new_pos, this_key, u, T, pstar, depth-1, J_i, policy_i, gamma)

    def __J_pos__(self, *args, **kwargs):
        pos = parsekw(kwargs, 'pos', np.inf*np.ones(3))
        pstar = parsekw(kwargs, 'pstar', np.inf*np.ones(3))
        if (pos == np.inf).any():
            print("No point specified, can't calculate cost")
            return False
        if (pstar == np.inf).any():
            print("No goal specified, can't calculate cost")
            return False
        return np.sqrt(np.sum((pos[0:2] - pstar[0:2])**2))

    def __J_vel__(self, *args, **kwargs):
        pos = parsekw(kwargs, 'pos', np.inf*np.ones(3))
        if len(pos) == 2:
            pos = np.append(pos, 0)
        pstar = parsekw(kwargs, 'pstar', np.inf*np.ones(3))
        jet = parsekw(kwargs, 'jet', False)
        if (pos == np.inf).any():
            print("No point specified, can't calculate cost")
            return False
        if (pstar == np.inf).any():
            print("No goal specified, can't calculate cost")
            return False
        return WindAwarePlanner.__cost_of_jetstream__(self, pos, pstar, jet)

    def __range_J__(self, cost_function, *args, **kwargs):
        pos = parsekw(kwargs, 'pos', np.inf*np.ones(3))
        std = parsekw(kwargs, 'std', np.inf*np.ones(3))
        pos = pos[0:2]
        std = std[0:2]
        J = np.zeros(5)
        p = np.zeros([5,len(pos)])
        p[0] = pos
        p[1] = pos+std
        p[2] = pos-std
        p[3] = pos+np.dot(std,np.array([1,-1]))
        p[4] = pos-np.dot(std,np.array([1,-1]))
        for i in range(len(p)):
            kwargs['pos'] = p[i]
            J[i] = cost_function(**kwargs)
        J_std = np.std(J)
        J_mu = J[0]
        return J_mu, J_std

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
        jets = self.jets.jetstreams.values()
        self.delta_p = np.zeros([len(jets),len(jets),2])
        self.delta_std = np.zeros([len(jets),len(jets),2])
        N = 500
        x_test = np.zeros(N)
        y_test = np.zeros(N)
        for i, key in enumerate(self.jets.jetstreams.keys()):
            self.jets.jetstreams[key].set_id(i)
        for i, jet1 in enumerate(self.jets.jetstreams.values()):
            for j, jet2 in enumerate(self.jets.jetstreams.values()):
                if i == j:
                    z_test = jet1.avg_alt
                    p_test = np.array([0, 0, z_test])
                    vx, vy, std_x, std_y = self.predict(p_test)
                    d_std = np.array([std_x, std_y]).reshape(2)
                    self.delta_p[jet1.id,jet2.id] = np.zeros(2)
                    self.delta_p[jet2.id,jet1.id] = np.zeros(2)
                    self.delta_std[jet1.id,jet2.id] = d_std
                    self.delta_std[jet2.id,jet1.id] = d_std
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
                    self.delta_p[jet1.id,jet2.id] = dp
                    self.delta_p[jet2.id,jet1.id] = dp
                    self.delta_std[jet1.id,jet2.id] = dp_std
                    self.delta_std[jet2.id,jet1.id] = dp_std
                    # print(d_std)

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
        min_leaf = self.__min_leaf__()
        best_pol = []
        stem = self.backedges[best_leaf][0]
        while True:
            pos = self.backedges[stem][1]
            best_pol = np.append(pos, best_pol)
            if self.backedges[stem][0] == stem:
                break
            stem = self.backedges[stem][0]
        return best_pol

    def __pol__(self, *args, **kwargs):
        leaf = parsekw(kwargs, 'leaf', None)
        pos = self.backedges[leaf][1]
        cost = self.backedges[leaf][2]
        pol = np.array(pos)
        J = np.array(cost)
        stem = self.backedges[leaf][0]
        while True:
            pos = self.backedges[stem][1]
            cost = self.backedges[stem][2]
            pol = np.append(pos, pol)
            J = np.append(cost, J)
            if self.backedges[stem][0] == stem:
                break
            stem = self.backedges[stem][0]
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

    def __cost_of_vel__(self, pos, pstar, vel):
        """
        Calculate the terminal cost of the given jetstream at the given position
        for the given set point.

        parameter pos position at which to calculate cost of jetstream
        parameter pstar set point/goal position
        parameter jet jetstream for which to calculate cost
        return terminal cost of jet at pos for pstar
        """

        # SETTING UP/INITIALIZATION
        # Get the balloon's current position and store its lateral and vertical
        # positions separately
        p = pos
        # Calculate the components of the wind velocity and store it as the
        # time derivative of the balloon's position
        vx = vel[0]
        vy = vel[1]
        pdot = np.squeeze(np.array([vx, vy]).T)
        pdothat = pdot / np.linalg.norm(pdot)
        # Calculate the unit vector along the balloon's lateral position vector
        norm_p = np.linalg.norm(p)
        # phat = p / norm_p if norm_p > 0 else p
        # Get the lateral position of the goal point
        pstar = pstar[0:2]
        # Translate the coordinate system such that pstar is at the origin
        phi = p - pstar
        norm_phi = np.linalg.norm(phi)
        phihat = phi / norm_phi if norm_phi > 0 else phi
        # CALCULATE COST:
        J_velocity = (np.dot(phihat, pdothat)+1)
        # Return the calculated cost
        return J_velocity

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
