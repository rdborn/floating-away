import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng

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
        self.__generate_dict__(2)
        self.__cluster__(threshold)
        self.__identify_streams__(streamsize)
        self.__classify_streams__()

    def __generate_dict__(self, key_idx):
        self.data = dict()
        for x in self.X:
            val = []
            for i in range(len(x)):
                if i != key_idx:
                    val = np.append(val, x[i])
            self.data[x[key_idx]] = np.array(val)

    def __cluster__(self, threshold):
        self.clusters = dict()
        self.cluster_labels = np.zeros([len(self.data.keys()),len(self.X[0])+1])
        self.n_clusters = 0
        cluster = []
        cluster_vals = []
        i = 0
        for key in np.sort(self.data.keys()):
            val = self.data[key]
            mag_val = val[0]
            dir_val = val[1]
            if np.var(np.append(cluster_vals,dir_val)) > threshold:
                self.clusters[self.n_clusters] = np.array(cluster)
                self.n_clusters += 1
                cluster = [key]
                cluster_vals = [dir_val]
            else:
                cluster = np.append(cluster, key)
                cluster_vals = np.append(cluster_vals, dir_val)
            self.cluster_labels[i] = np.array([key, mag_val, dir_val, self.n_clusters])
            i += 1

    def __identify_streams__(self, streamsize):
        jetstreams = []
        for c in self.clusters.keys():
            if len(self.clusters[c]) > streamsize:
                jetstreams = np.append(jetstreams,c)
        self.stream_ids = np.array(jetstreams)

    def __classify_streams__(self):
        alt_avgs = np.zeros(len(self.stream_ids))
        mag_avgs = np.zeros(len(self.stream_ids))
        dir_avgs = np.zeros(len(self.stream_ids))
        n = np.zeros(len(self.stream_ids))
        for x in self.cluster_labels:
            idx = (self.stream_ids == x[3])
            alt_avgs[idx] += x[0]
            mag_avgs[idx] += x[1]
            dir_avgs[idx] += x[2]
            n += idx
        alts = alt_avgs / n
        mags = mag_avgs / n
        dirs = dir_avgs / n
        self.jetstreams = dict()
        for stream in self.stream_ids:
            idx = (self.stream_ids == stream)
            self.jetstreams[stream] = np.array([alts[idx], mags[idx], dirs[idx]])

    def plot(self):
        print(self.n_clusters)
        f, (a1, a2, a3) = plt.subplots(1, 3, sharey=True)
        lim = 1.1
        a1.scatter( self.cluster_labels[:,3], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')
        a1.scatter( self.stream_ids, 1.1*np.ones(len(self.stream_ids)))
        a2.scatter( self.cluster_labels[:,2], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')
        a3.scatter( self.cluster_labels[:,1], self.cluster_labels[:,0], c = self.cluster_labels[:,3]%9, cmap='Set1')
        a2.set_xlim([-lim,lim])
        a3.set_xlim([-lim,lim])
