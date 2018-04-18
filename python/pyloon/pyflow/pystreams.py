import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from cvxopt import matrix, solvers

from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng
import pyutils.pyutils as pyutils

class JetStream:
    def __init__(self, *args, **kwargs):
        self.avg_alt = parsekw(kwargs, 'avg_alt', 0)
        self.min_alt = parsekw(kwargs, 'min_alt', 0)
        self.max_alt = parsekw(kwargs, 'max_alt', 0)
        self.magnitude = parsekw(kwargs, 'magnitude', 0)
        self.direction = parsekw(kwargs, 'direction', 0)
        vx = self.magnitude * np.cos(direction)
        vy = self.magnitude * np.sin(direction)
        v = np.array([vx, vy])
        self.v = v

    def set_id(self, id):
        self.id = id

    def ride_for_dt(self, dt):
        vx = self.magnitude * np.cos(self.direction)
        vy = self.magnitude * np.sin(self.direction)
        vz = 0.0
        v = np.array([vx, vy, vz])
        dp = dt * v
        return dp

class JetStreamIdentifier:
    def __init__(self, *args, **kwargs):
        self.X = parsekw(kwargs, 'data', None)

    def __smooth__(self, i, w):
        xnew = np.zeros(len(self.X[:,i]))
        for j in range(len(xnew)):
            xsmooth = self.X[j,i]
            for k in range(w):
                if k > 0:
                    if j + k < len(xnew):
                        xsmooth += self.X[j+k,i]
                    if i - j >= 0:
                        xsmooth += self.X[j-k,i]
            xsmooth = xsmooth / (w + 1)
            xnew[j] = xsmooth
        self.X[:,i] = xnew

    def __str__(self):
        return_str = "\tJetstreams (min, avg, max alts, mag, dir):\n"
        for jet in self.jetstreams.values():
            return_str = return_str + "\t\t" + str(np.int(jet.min_alt)) + \
                "\t" + str(np.int(jet.avg_alt)) + \
                "\t" + str(np.int(jet.max_alt)) + \
                "\t" + str(np.int(jet.magnitude)) + \
                "\t" + str(np.int(jet.direction * 180.0 / np.pi)) + "\n"
        return return_str

    def spanning_quality(self):
        vx = np.zeros(len(self.jetstreams.keys()))
        vy = np.zeros(len(self.jetstreams.keys()))
        if len(vx) == 0:
            return 1.
        for i, jet in enumerate(self.jetstreams.values()):
            direction = jet.direction
            vx[i] = np.cos(direction)
            vy[i] = np.sin(direction)
        A = np.matrix([vx, vy])
        C = np.matrix(np.ones(A.shape[1]))
        p = np.matrix(np.eye(A.shape[1]))
        y = np.matrix([0.,0.]).T
        I = np.matrix(np.eye(A.shape[1]))
        Z = np.matrix(np.zeros(A.shape[1])).T

        _P = matrix(2 * (A - y * C).T * (A - y * C))
        _q = matrix((0 * C).T)
        _G = matrix(-I)
        _h = matrix(Z)
        _A = None
        _b = None

        sol = solvers.qp(_P, _q, _G, _h, _A, _b)
        x = np.squeeze(np.array(sol['x']))
        x /= np.sum(x)

        return (np.dot(x,vx)**2 + np.dot(x,vy)**2)

""" NOT SUPPORTED """
class ClusterIdentifier(JetStreamIdentifier):
    def __init__(self, *args, **kwargs):
        JetStreamIdentifier.__init__(self, data=kwargs.get('data'))
        self.B = parsekw(kwargs, 'B', 10)
        self.__smooth__(1, 20)
        n_clusters = self.__cluster__(kmax=kwargs.get('kmax'))
        self.kmeans = KMeans(n_clusters=n_clusters).fit(self.X)
        pass

    def streams(self):
        return self.kmeans.cluster_centers_

    def predict(self, x):
        stream = self.kmeans.predict(x)
        return self.kmeans.cluster_centers_[stream]

    def __cluster__(self, *args, **kwargs):
        kmax = parsekw(kwargs, 'kmax', 1)
        gap_prev = 0.0
        for k in range(kmax):
            flag = True
            if flag:
                n_clusters = k + 1
                gap, s = self.__gap_stat__(n_clusters=n_clusters)
                if gap_prev - gap - s > 0:
                    return k
        print("Optimal clustering not found. Returning kmax.")
        return kmax

    # Adapted from
    #   https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
    # Algorithm from
    #   Tibshirani et al, "Estimating the number of clusters in a data set via the gap statistic"
    def __gap_stat__(self, *args, **kwargs):
        n_clusters = parsekw(kwargs, 'n_clusters', 1)
        kmeans = KMeans(n_clusters=n_clusters).fit(self.X)
        Wks = self.__log_wk__(kmeans.inertia_, self.X)
        BWkbs = np.zeros(self.B)
        for i in range(self.B):
            Xb = self.__ref_data__()
            bkmeans = KMeans(n_clusters=n_clusters).fit(Xb)
            BWkbs[i] = self.__log_wk__(bkmeans.inertia_, Xb)
        Wkbs = np.sum(BWkbs) / self.B
        gap = np.sum(BWkbs - Wks) / self.B
        sdk = np.sqrt(np.sum((BWkbs - Wkbs)**2) / self.B)
        sk = sdk * np.sqrt(1 + 1 / self.B)
        return gap, sk

    def __log_wk__(self, inertia, X):
        return np.log(inertia / (2 * len(X)))

    def __ref_data__(self):
        xmin, xmax = self.__bounding_box__()
        Xb = np.zeros([len(self.X), len(xmin)])
        for i in range(len(Xb)):
            xnew = np.zeros(len(xmin))
            for j in range(len(xnew)):
                xnew[j] = np.random.uniform(xmin[j], xmax[j])
            Xb[i] = xnew
        return Xb

    def __bounding_box__(self, *args, **kwargs):
        xmin = np.min(self.X, axis=0)
        xmax = np.max(self.X, axis=0)
        return xmin, xmax

    def plot(self):
        lim = 1.1
        f, (a1, a2) = plt.subplots(1, 2, sharey=True)
        a1.scatter( self.X[:,0], self.X[:,2], c=self.kmeans.labels_)
        a1.set_xlim([-lim,lim])
        a2.scatter( self.X[:,1], self.X[:,2], c=self.kmeans.labels_)
        a2.set_xlim([-1.1,1.1])
        print(self.kmeans.labels_)

class VarThresholdIdentifier(JetStreamIdentifier):
    def __init__(self, *args, **kwargs):
        threshold = parsekw(kwargs, 'threshold', 0.005)
        streamsize = parsekw(kwargs, 'streamsize', 15)
        JetStreamIdentifier.__init__(self, data=kwargs.get('data'))
        self.__generate_dict__(2)               # store data in a dictionary indexed by altitude
        self.__cluster__(threshold)             # cluster data into sets of low variance
        self.__identify_streams__(streamsize)   # identify large clusters
        self.__classify_streams__()             # assign each stream an average altitude, magnitude, and direction

    def __generate_dict__(self, key_idx):
        # Initialize data structure as dictionary
        self.data = dict()
        # For every data point in the raw data...
        for x in self.X:
            # Initialize an empy list
            val = []
            # For every element in the data point
            # (i.e. magnitude, direction, altitude)...
            for i in range(len(x)):
                # If this element is not the one we want to use
                # for the dictionary keys...
                if i != key_idx:
                    # Add this element to val
                    val = np.append(val, x[i])
            # Add this data point to the dictionary,
            # indexed by the appropriate value
            self.data[x[key_idx]] = np.array(val)

    def __cluster__(self, threshold):
        # Initialize the clusters data structure as a dictionary
        self.clusters = dict()
        # Initialize the cluster_labels data structure as an array of zeros
        # with the same number of rows as data points and the same number
        # of columns as each data point
        self.cluster_labels = np.zeros([len(self.data.keys()),len(self.X[0])+1-1])
        # Initialize the total number of clusters we have created to zero
        self.n_clusters = 0
        # Initialize the cluster data structure to an empty list
        cluster = []
        # Initialize the cluster_values data structure to an empty list
        cluster_vals = []
        # For each altitude value (in ascending order)...
        keys = np.sort(self.data.keys())
        mag_vals = np.zeros(len(keys))
        dir_vals = np.zeros(len(keys))
        alt_vals = np.zeros(len(keys))
        unbounded = np.zeros(len(keys))
        for i, key in enumerate(keys):
            val = self.data[key]
            mag_vals[i] = val[0]
            dir_vals[i] = val[1]
            alt_vals[i] = key
            unbounded[i] = val[2]
        dir_vals = pyutils.continuify_angles(dir_vals * 180.0 / np.pi) * np.pi / 180.0
        for i in range(len(dir_vals)):
            # If adding the value of the wind's direction at this altitude
            # to the set of directions at each altitude sampled so far in this
            # cluster will put the variance of the sample over the threshold,
            # OR if we are at the end of our data...
            at_the_end = (i == len(dir_vals)-1)
            too_wild = (np.var(np.append(cluster_vals,dir_vals[i])) > threshold)
            if unbounded[i] or too_wild or at_the_end:
                # Store the altitudes sampled in this cluster into the clusters
                # dictionary, indexed by the order in which we added the clusters
                self.clusters[self.n_clusters] = np.array(cluster)
                # Increment the number of clusters we have identified
                self.n_clusters += 1
                # Reset the local variable cluster to hold only the most recent
                # altitude sampled
                cluster = [alt_vals[i]]
                # Reset the local variable cluster_values to hold only the most
                # recent direction sampled
                cluster_vals = [dir_vals[i]]
            else:
                # Add the most recently sampled altitude to the local variable cluster
                cluster = np.append(cluster, alt_vals[i])
                # Add the most recently sampled direction to the local variable cluster_values
                cluster_vals = np.append(cluster_vals, dir_vals[i])
            # Label this data point as belonging to this cluster.
            # cluster_labels holds the altitude, magnitude, direction, and
            # cluster ID of each data point in no particular order
            self.cluster_labels[i] = np.array([alt_vals[i], mag_vals[i], dir_vals[i], self.n_clusters])

    def __identify_streams__(self, streamsize):
        # Initialize jetstreams to an empty list
        jetstreams = []
        # For each cluster...
        for c in self.clusters.keys():
            # If the cluster is large enough...
            if len(self.clusters[c]) > streamsize:
                # Add the cluster to the jetstreams
                jetstreams = np.append(jetstreams,c)
        # Store the jetstreams we identified
        self.stream_ids = np.array(jetstreams)

    def __classify_streams__(self):
        # Initialize the summed altitude, magnitude, and direction data structures
        # as empty arrays with the same length as number of jetstreams
        alt_sum = np.zeros(len(self.stream_ids))
        mag_sum = np.zeros(len(self.stream_ids))
        dir_sum = np.zeros(len(self.stream_ids))
        # Initialize the number of samples in each jetstream to an empty array
        # with the same length as number of jetstreams
        n = np.zeros(len(self.stream_ids))
        # Initialize the maximum and minimum altitudes data structures as empty
        # arrays with the same length as number of jetstreams
        alt_min = np.inf * np.ones(len(self.stream_ids))
        alt_max = -np.inf * np.ones(len(self.stream_ids))
        # For each data point...
        for x in self.cluster_labels:
            # Identify which cluster the data point belongs to
            idx = (self.stream_ids == x[3])
            # Add the data point's values to the appropriate averages
            alt_sum[idx] += x[0]
            mag_sum[idx] += x[1]
            dir_sum[idx] += np.cos(x[2])
            # Increment the appropriate number of samples
            n[idx] += 1
            # If necessary, update the appropriate maximums and minimums
            if x[0] > alt_max[idx]:
                alt_max[idx] = x[0]
            if x[0] < alt_min[idx]:
                alt_min[idx] = x[0]
        # Average the altitudes, magnitudes, and directions of each jetstream
        alt_avg = alt_sum / n
        mag_avg = mag_sum / n
        dir_avg = dir_sum / n
        theta = np.zeros(len(self.stream_ids))
        for i, cos_theta in enumerate(dir_avg):
            theta_a = np.arccos(cos_theta)
            theta_b = 2*np.pi - theta_a
            for x in self.cluster_labels:
                if x[3] == self.stream_ids[i]:
                    if np.cos(theta_a - x[2]) > np.cos(theta_b - x[2]):
                        theta[i] = theta_a
                        break
                    else:
                        theta[i] = theta_b
                        break

        # Initialize the jetstreams data structure as an empty dicitonary
        self.jetstreams = dict()
        # For each jetstream...
        for i, stream in enumerate(self.stream_ids):
            # Store the jetstream object indexed by its ID
            self.jetstreams[stream] = JetStream(avg_alt=alt_avg[i],
                                                min_alt=alt_min[i],
                                                max_alt=alt_max[i],
                                                magnitude=mag_avg[i],
                                                direction=theta[i])

    def find(self, z):
        # Get all the jetstreams
        jets = self.jetstreams.values()
        # For each jetstream...
        for jet in jets:
            # If the test point falls within the jetstream's altitude bounds...
            if z > jet.min_alt and z < jet.max_alt:
                # Return this jetstream
                return jet
        print("Jetstream not found. Can't return anything.")
        return_jet = JetStream( avg_alt=0,
                                min_alt=0,
                                max_alt=0,
                                magnitude=0,
                                direction=0)
        return_jet.set_id(-1)
        return return_jet

    def find_adjacent(self, z, search_dir):
        jets = self.jetstreams.values()
        this_jetstream = self.find(z)
        adjacent_jetstream = self.find(z)
        min_gap_size = np.inf
        if search_dir >= 0:
            this_max_alt = this_jetstream.max_alt
            for jet in jets:
                gap_size = jet.min_alt - this_max_alt
                if gap_size > 0 and gap_size < min_gap_size:
                    min_gap_size = gap_size
                    adjacent_jetstream = jet
            if adjacent_jetstream.max_alt == this_max_alt:
                print("No adjacent jetstream in requested direction. Returning current jetstream")
        else:
            this_min_alt = this_jetstream.min_alt
            for jet in jets:
                gap_size = this_min_alt - jet.max_alt
                if gap_size > 0 and gap_size < min_gap_size:
                    min_gap_size = gap_size
                    adjacent_jetstream = jet
            if adjacent_jetstream.min_alt == this_min_alt:
                print("No adjacent jetstream in requested direction. Returning current jetstream")
        return adjacent_jetstream

    def plot(self):
        print(self.n_clusters)
        f, (a1, a2, a3) = plt.subplots(1, 3, sharey=True)
        a1.scatter( self.cluster_labels[:,3], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')
        a1.scatter( self.stream_ids, -100.0*np.ones(len(self.stream_ids)))
        a2.scatter( self.cluster_labels[:,2], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')
        a3.scatter( self.cluster_labels[:,1], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')

class VarThresholdIdentifier2(JetStreamIdentifier):
    def __init__(self, *args, **kwargs):
        threshold = parsekw(kwargs, 'threshold', 0.005)
        streamsize = parsekw(kwargs, 'streamsize', 15)
        expectation = parsekw(kwargs, 'expectation', False)
        self.__generate_dict__(**kwargs)               # store data in a dictionary indexed by altitude
        if expectation:
            self.__cluster__(threshold)             # cluster data into sets of low variance
        else:
            self.__cluster__(threshold * 10)             # cluster data into sets of low variance
        self.__identify_streams__(streamsize)   # identify large clusters
        self.__classify_streams__()             # assign each stream an average altitude, magnitude, and direction

    def __generate_dict__(self, *args, **kwargs):
        vx = parsekw(kwargs, 'vx', None)
        vy = parsekw(kwargs, 'vy', None)
        stdx = parsekw(kwargs, 'stdx', None)
        stdy = parsekw(kwargs, 'stdy', None)
        alt = parsekw(kwargs, 'alt', None)
        expectation = parsekw(kwargs, 'expectation', False)
        # Initialize data structure as dictionary
        self.data = dict()
        # For every data point in the raw data...
        for i in range(len(vx)):
            magnitude = np.sqrt(vx[i]**2 + vy[i]**2)
            dir_nom = np.arctan2(vy[i], vx[i])
            if expectation:
                std_dir = 1e-6
            else:
                # dir_lower, dir_nom, dir_upper = pyutils.get_angle_range(vx[i], vy[i], stdx[i], stdy[i])
                # std_dir = (dir_upper - dir_nom) * np.pi / 180.0
                # dir_nom = dir_nom * np.pi / 180.0
                n_dir_samples = 100
                dir_nom, std_dir = pyutils.get_angle_dist(vx[i], vy[i], stdx[i], stdy[i], n_dir_samples)
            val = [magnitude, dir_nom, std_dir]
            self.data[alt[i]] = np.array(val)

    def __cluster__(self, threshold):
        # Initialize the clusters data structure as a dictionary
        self.clusters = dict()
        # Initialize the cluster_labels data structure as an array of zeros
        # with the same number of rows as data points and the same number
        # of columns as each data point
        self.cluster_labels = np.zeros([len(self.data.keys()), 4])
        # Initialize the total number of clusters we have created to zero
        self.n_clusters = 0
        # Initialize the cluster data structure to an empty list
        cluster = []
        # Initialize the cluster_values data structure to an empty list
        cluster_vals = []
        # For each altitude value (in ascending order)...
        keys = np.sort(self.data.keys())
        mag_vals = np.zeros(len(keys))
        dir_vals = np.zeros(len(keys))
        alt_vals = np.zeros(len(keys))
        std_vals = np.zeros(len(keys))
        for i, key in enumerate(keys):
            val = self.data[key]
            mag_vals[i] = val[0]
            dir_vals[i] = val[1]
            alt_vals[i] = key
            std_vals[i] = val[2]
        dir_vals = pyutils.continuify_angles(dir_vals * 180.0 / np.pi) * np.pi / 180.0
        # additional_var = 0.0
        for i in range(len(dir_vals)):
            # If adding the value of the wind's direction at this altitude
            # to the set of directions at each altitude sampled so far in this
            # cluster will put the variance of the sample over the threshold,
            # OR if we are at the end of our data...
            at_the_end = (i == len(dir_vals)-1)
            # additional_var += std_vals[i]**2
            n_dir_samples = 100
            dir_samples = np.random.normal(dir_vals[i], std_vals[i], n_dir_samples)
            too_wild = (np.var(np.append(cluster_vals,dir_samples)) > threshold)
            if too_wild or at_the_end:
                # Store the altitudes sampled in this cluster into the clusters
                # dictionary, indexed by the order in which we added the clusters
                self.clusters[self.n_clusters] = np.array(cluster)
                # Increment the number of clusters we have identified
                self.n_clusters += 1
                # Reset the local variable cluster to hold only the most recent
                # altitude sampled
                cluster = [alt_vals[i]]
                # Reset the local variable cluster_values to hold only the most
                # recent direction sampled
                cluster_vals = [dir_samples]
                # additional_var = 0.0
            else:
                # Add the most recently sampled altitude to the local variable cluster
                cluster = np.append(cluster, alt_vals[i])
                # Add the most recently sampled direction to the local variable cluster_values
                cluster_vals = np.append(cluster_vals, dir_samples)
            # Label this data point as belonging to this cluster.
            # cluster_labels holds the altitude, magnitude, direction, and
            # cluster ID of each data point in no particular order
            self.cluster_labels[i] = np.array([alt_vals[i], mag_vals[i], dir_vals[i], self.n_clusters])

    def __identify_streams__(self, streamsize):
        # Initialize jetstreams to an empty list
        jetstreams = []
        # For each cluster...
        for c in self.clusters.keys():
            # If the cluster is large enough...
            if len(self.clusters[c]) > streamsize:
                # Add the cluster to the jetstreams
                jetstreams = np.append(jetstreams,c)
        # Store the jetstreams we identified
        self.stream_ids = np.array(jetstreams)

    def __classify_streams__(self):
        # Initialize the summed altitude, magnitude, and direction data structures
        # as empty arrays with the same length as number of jetstreams
        alt_sum = np.zeros(len(self.stream_ids))
        mag_sum = np.zeros(len(self.stream_ids))
        dir_sum = np.zeros(len(self.stream_ids))
        # Initialize the number of samples in each jetstream to an empty array
        # with the same length as number of jetstreams
        n = np.zeros(len(self.stream_ids))
        # Initialize the maximum and minimum altitudes data structures as empty
        # arrays with the same length as number of jetstreams
        alt_min = np.inf * np.ones(len(self.stream_ids))
        alt_max = -np.inf * np.ones(len(self.stream_ids))
        # For each data point...
        for x in self.cluster_labels:
            # Identify which cluster the data point belongs to
            idx = (self.stream_ids == x[3])
            # Add the data point's values to the appropriate averages
            alt_sum[idx] += x[0]
            mag_sum[idx] += x[1]
            dir_sum[idx] += np.cos(x[2])
            # Increment the appropriate number of samples
            n[idx] += 1
            # If necessary, update the appropriate maximums and minimums
            if x[0] > alt_max[idx]:
                alt_max[idx] = x[0]
            if x[0] < alt_min[idx]:
                alt_min[idx] = x[0]
        # Average the altitudes, magnitudes, and directions of each jetstream
        alt_avg = alt_sum / n
        mag_avg = mag_sum / n
        dir_avg = dir_sum / n
        theta = np.zeros(len(self.stream_ids))
        for i, cos_theta in enumerate(dir_avg):
            theta_a = np.arccos(cos_theta)
            theta_b = 2*np.pi - theta_a
            for x in self.cluster_labels:
                if x[3] == self.stream_ids[i]:
                    if np.cos(theta_a - x[2]) > np.cos(theta_b - x[2]):
                        theta[i] = theta_a
                        break
                    else:
                        theta[i] = theta_b
                        break

        # Initialize the jetstreams data structure as an empty dicitonary
        self.jetstreams = dict()
        # For each jetstream...
        for i, stream in enumerate(self.stream_ids):
            # Store the jetstream object indexed by its ID
            self.jetstreams[stream] = JetStream(avg_alt=alt_avg[i],
                                                min_alt=alt_min[i],
                                                max_alt=alt_max[i],
                                                magnitude=mag_avg[i],
                                                direction=theta[i])

    def find(self, z):
        # Get all the jetstreams
        jets = self.jetstreams.values()
        # For each jetstream...
        for jet in jets:
            # If the test point falls within the jetstream's altitude bounds...
            if z > jet.min_alt and z < jet.max_alt:
                # Return this jetstream
                return jet
        print("Jetstream not found. Can't return anything.")
        return_jet = JetStream( avg_alt=0,
                                min_alt=0,
                                max_alt=0,
                                magnitude=0,
                                direction=0)
        return_jet.set_id(-1)
        return return_jet

    def find_adjacent(self, z, search_dir):
        jets = self.jetstreams.values()
        this_jetstream = self.find(z)
        adjacent_jetstream = self.find(z)
        min_gap_size = np.inf
        if search_dir >= 0:
            this_max_alt = this_jetstream.max_alt
            for jet in jets:
                gap_size = jet.min_alt - this_max_alt
                if gap_size > 0 and gap_size < min_gap_size:
                    min_gap_size = gap_size
                    adjacent_jetstream = jet
            if adjacent_jetstream.max_alt == this_max_alt:
                print("No adjacent jetstream in requested direction. Returning current jetstream")
        else:
            this_min_alt = this_jetstream.min_alt
            for jet in jets:
                gap_size = this_min_alt - jet.max_alt
                if gap_size > 0 and gap_size < min_gap_size:
                    min_gap_size = gap_size
                    adjacent_jetstream = jet
            if adjacent_jetstream.min_alt == this_min_alt:
                print("No adjacent jetstream in requested direction. Returning current jetstream")
        return adjacent_jetstream

    def plot(self):
        print(self.n_clusters)
        f, (a1, a2, a3) = plt.subplots(1, 3, sharey=True)
        a1.scatter( self.cluster_labels[:,3], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')
        a1.scatter( self.stream_ids, -100.0*np.ones(len(self.stream_ids)))
        a2.scatter( self.cluster_labels[:,2], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')
        a3.scatter( self.cluster_labels[:,1], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')
