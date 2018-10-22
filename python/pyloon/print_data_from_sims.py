import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable as sm
from scipy.interpolate import griddata
from scipy.stats import norm
import os.path
import sys
from pyutils import pyutils
from optiloon.pycosts import J_position as Jp
from optiloon.pycosts import J_velocity as Jv
from matplotlib.patches import Circle, Ellipse
from matplotlib.collections import PatchCollection

verbose = False

def extract_data_no_fields(file):
    data = []
    with open(file) as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i < 4:
                i += 1
            else:
                data.append(row)
    return data

def process(data, m):
    processed = []
    i = 0
    for d in data:
        t = []
        x = []
        y = []
        vx = []
        vy = []
        jp = []
        for row in d:
            ti = float(row[0])
            xi = float(row[1])
            yi = float(row[2])
            vxi = float(row[5][1:-1])
            vyi = float(row[6][1:-1])
            jpi = Jp(p=np.array([xi, yi], dtype=float), pstar=np.zeros(2))
            t.append(ti)
            x.append(xi)
            y.append(yi)
            vx.append(vxi)
            vy.append(vyi)
            jp.append(jpi)
        processed_i = [t, x, y, vx, vy, jp]
        processed.append(processed_i)
        i += 1
        if verbose:
            print(str(i) + "/" + str(m))
        if i >= m:
            break
    return processed

def extract(parent_dir, case_no, depth, n, we_should_grid):
    path = parent_dir \
        + "/case-" + str(case_no) \
        + "/depth-" + str(depth) + "/"
    if verbose:
        print path
    directories = glob(path + "*/")
    raw_data = []
    for d in directories:
        if verbose:
            print d
        f = d + d[len(path + 'fig_'):-1] + '.csv'
        raw_data.append(extract_data_no_fields(f))
    m = len(directories)
    if verbose:
        print "processing"
    data = process(raw_data, m)

    if we_should_grid:
        T = 45.*3600
        N = 1000
        data = pyutils.grid_uneven_data(data, 0, T, N)

    return data

def prow(label, values):
    msg = "{:10}".format(label) + "|"
    divider = "\n-----------"
    for v in values:
        msg = msg + "{:10.2f}".format(v) + " |"
        divider = divider + "------------"
    msg = msg + divider
    print(msg)
    return msg

def phead(head, length):
    msg = ("{:*^" + str(length) + "}").format("")
    msg = msg + "\n" + ("{:*^" + str(length) + "}").format(" " + head + " ")
    msg = msg + "\n" + ("{:*^" + str(length) + "}").format("")
    print(msg)
    return msg

def very_specific_extraction_function(data, i, idx_last, idx):
    val = data[i]
    val[np.isnan(val)] = 0.
    val = val[idx_last:idx]
    return val

def pdata0(cases, depths, labels, n, plotting):
    parent_dir = "./datasets"
    if plotting:
        fig = plt.figure()
        fig.set_figheight(10.0)
        fig.set_figwidth(7.0)
        n_plots = 6
        gs = gridspec.GridSpec(n_plots, 1)
        axes = []
        for i in range(n_plots):
            if len(axes) > 0:
                axes.append(fig.add_subplot(gs[i:(i+1),:], sharex=axes[0]))
                if len(axes) < n_plots:
                    plt.setp(axes[i].get_xticklabels(), visible=False)
            else:
                axes.append(fig.add_subplot(gs[i:(i+1),:]))
                plt.setp(axes[i].get_xticklabels(), visible=False)
        axes[0].set_xlim([0, 45])
        axes[0].set_ylim([-100,100])
        axes[1].set_ylim([-100,100])
        axes[2].set_ylim([0,100])
        axes[3].set_ylim([0,70])
        axes[4].set_ylim([0,50])
        axes[5].set_ylim([0,50])
        axes[0].set_ylabel('x (km)')
        axes[1].set_ylabel('y (km)')
        axes[2].set_ylabel('pos error (km)')
        axes[3].set_ylabel('std(x) (km)')
        axes[4].set_ylabel('std(y) (km)')
        axes[5].set_ylabel('std(error) (km)')
        axes[5].set_xlabel('time (hr)')
    for i in range(len(cases)):
        case_no = cases[i]
        depth = depths[i]
        data = extract(parent_dir, case_no, depth, n, True)
        T = 45.*3600
        N = 1000
        t_last = 0.
        idx_last = 0
        t_window = T / n
        tidx = 0
        xidx = 2
        yidx = 1
        jidx = 5
        idx = 0
        mu_x = []
        mu_y = []
        mu_jp = []
        min_jp = []
        max_jp = []
        sigma_x = []
        sigma_y = []
        sigma_jp = []
        times = []
        t = data[0][tidx]
        for j in range(n):
            while t[idx] - t_last < t_window and idx < N - 1:
                idx += 1
            xi = []
            yi = []
            jpi = []
            for d in data:
                xi.extend(very_specific_extraction_function(d, xidx, idx_last, idx)*1e-3)
                yi.extend(very_specific_extraction_function(d, yidx, idx_last, idx)*1e-3)
                jpi.extend(very_specific_extraction_function(d, jidx, idx_last, idx)*1e-3)
            mu_x.append(np.mean(xi))
            mu_y.append(np.mean(yi))
            mu_jp.append(np.mean(jpi))
            min_jp.append(np.min(jpi))
            max_jp.append(np.max(jpi))
            sigma_x.append(np.std(xi))
            sigma_y.append(np.std(yi))
            sigma_jp.append(np.std(jpi))
            t_last = t[idx]
            times.append(t_last / 3600)
            idx_last = idx
        phead(labels[i], 11+12*len(times))
        prow('t', times)
        prow('mu_x', mu_x)
        prow('mu_y', mu_y)
        prow('mu_jp', mu_jp)
        prow('min_jp', min_jp)
        prow('max_jp', max_jp)
        prow('sigma_x', sigma_x)
        prow('sigma_y', sigma_y)
        prow('sigma_jp', sigma_jp)
        if plotting:
            tplot = np.array(times)
            tprev = 0.
            for k in range(len(tplot)):
                tcurr = tplot[k]
                tplot[k] = (tcurr + tprev) / 2.
                tprev = tcurr
            axes[0].plot(tplot, mu_x)
            axes[1].plot(tplot, mu_y)
            axes[2].plot(tplot, mu_jp)
            axes[2].fill_between(tplot, np.array(mu_jp)+2*np.array(sigma_jp), np.array(mu_jp)-2*np.array(sigma_jp), alpha=0.5)
            axes[3].plot(tplot, sigma_x)
            axes[4].plot(tplot, sigma_y)
            axes[5].plot(tplot, sigma_jp)
    if plotting:
        plt.show()


cases = ['2a', '2b']
depths = [4, 4]
labels = cases#['2a', '2b', '2m']
n = 1
pdata0(cases, depths, labels, n, False)
