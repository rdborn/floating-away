import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng

class JetStream:
    def __init__(self, *args, **kwargs):
        self.avg_alt = parsekw(kwargs, 'avg_alt', 0)
        self.min_alt = parsekw(kwargs, 'min_alt', 0)
        self.max_alt = parsekw(kwargs, 'max_alt', 0)
        self.magnitude = parsekw(kwargs, 'magnitude', 0)
        self.direction = parsekw(kwargs, 'direction', 0)

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
        self.cluster_labels = np.zeros([len(self.data.keys()),len(self.X[0])+1])
        # Initialize the total number of clusters we have created to zero
        self.n_clusters = 0
        # Initialize the cluster data structure to an empty list
        cluster = []
        # Initialize the cluster_values data structure to an empty list
        cluster_vals = []
        # For each altitude value (in ascending order)...
        for i, key in enumerate(np.sort(self.data.keys())):
            # Retrieve the wind magnitude and direction at this altitude
            val = self.data[key]
            mag_val = val[0]
            dir_val = val[1]
            # If adding the value of the wind's direction at this altitude
            # to the set of directions at each altitude sampled so far in this
            # cluster will put the variance of the sample over the threshold,
            # OR if we are at the end of our data...
            if np.var(np.append(cluster_vals,dir_val)) > threshold or i == len(self.data.keys())-1:
                # Store the altitudes sampled in this cluster into the clusters
                # dictionary, indexed by the order in which we added the clusters
                self.clusters[self.n_clusters] = np.array(cluster)
                # Increment the number of clusters we have identified
                self.n_clusters += 1
                # Reset the local variable cluster to hold only the most recent
                # altitude sampled
                cluster = [key]
                # Reset the local variable cluster_values to hold only the most
                # recent direction sampled
                cluster_vals = [dir_val]
            else:
                # Add the most recently sampled altitude to the local variable cluster
                cluster = np.append(cluster, key)
                # Add the most recently sampled direction to the local variable cluster_values
                cluster_vals = np.append(cluster_vals, dir_val)
            # Label this data point as belonging to this cluster.
            # cluster_labels holds the altitude, magnitude, direction, and
            # cluster ID of each data point in no particular order
            self.cluster_labels[i] = np.array([key, mag_val, dir_val, self.n_clusters])

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
        alt_min = np.zeros(len(self.stream_ids))
        alt_max = np.zeros(len(self.stream_ids))
        # For each data point...
        for x in self.cluster_labels:
            # Identify which cluster the data point belongs to
            idx = (self.stream_ids == x[3])
            # Add the data point's values to the appropriate averages
            alt_sum[idx] += x[0]
            mag_sum[idx] += x[1]
            dir_sum[idx] += x[2]
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
        # Initialize the jetstreams data structure as an empty dicitonary
        self.jetstreams = dict()
        # For each jetstream...
        for i, stream in enumerate(self.stream_ids):
            # Store the jetstream object indexed by its ID
            self.jetstreams[stream] = JetStream(avg_alt=alt_avg[i],
                                                min_alt=alt_min[i],
                                                max_alt=alt_max[i],
                                                magnitude=mag_avg[i],
                                                direction=np.arccos(dir_avg[i]))

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

    def plot(self):
        print(self.n_clusters)
        f, (a1, a2, a3) = plt.subplots(1, 3, sharey=True)
        a1.scatter( self.cluster_labels[:,3], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')
        a1.scatter( self.stream_ids, -100.0*np.ones(len(self.stream_ids)))
        a2.scatter( self.cluster_labels[:,2], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')
        a3.scatter( self.cluster_labels[:,1], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')