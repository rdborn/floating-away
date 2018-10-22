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

def get_dirs(directory, substr):
    subdirs = glob((directory + "*/"))
    dirs = []
    for d in subdirs:
        if substr in d:
            dirs.append(d)
    return dirs

def get_dirs_depth_n(directory, depth):
    substr = str(np.int(depth)) + "-depth"
    return get_dirs(directory, substr)

def get_dirs_n_samples(directory, n_samples):
    substr = str(np.int(n_samples)) + "-samples"
    return get_dirs(directory, substr)

def clean(str):
    return str.replace('[','').replace(']','')

def process(str):
    val = np.float(clean(str))
    val = 0.0 if np.isnan(val) else val
    return val

def extract_data(csvfile):
    # csvfile = directory + directory[len('./naives/fig_'):-1] + '.csv'
    if not os.path.isfile(csvfile):
        return -np.inf*np.ones(13)
    t = []
    xstar = []
    ystar = []
    zstar = []
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []
    jpos = []
    jvel = []
    jtot = []
    stdx = []
    stdy = []
    with open(csvfile) as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i < 4:
                i += 1
            else:
                t.append(process(row[0]))
                xstar.append(process(row[1]))
                ystar.append(process(row[2]))
                zstar.append(process(row[3]))
                x.append(process(row[4]))
                y.append(process(row[5]))
                z.append(process(row[6]))
                vx.append(process(row[7]))
                vy.append(process(row[8]))
                vz.append(process(row[9]))
                jpos.append(process(row[10]))
                jvel.append(process(row[11]))
                jtot.append(jpos[-1] + jvel[-1] * 1e5)
                # jtot.append(process(row[12]))
                if len(row) > 13:
                    stdx.append(process(row[13]))
                    stdy.append(process(row[14]))
    t = np.array(t)
    xstar = np.array(xstar)
    ystar = np.array(ystar)
    zstar = np.array(zstar)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)
    jpos = np.array(jpos)
    jvel = np.array(jvel)
    jtot = np.array(jtot)
    stdx = np.array(stdx)
    stdy = np.array(stdy)
    return t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot, stdx, stdy

def get_data_n_samples(parent_dir, n):
    directories = get_dirs_n_samples(parent_dir, n)
    t = []
    xstar = []
    ystar = []
    zstar = []
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []
    jpos = []
    jvel = []
    jtot = []
    stdx = []
    stdy = []
    print("Extracting from...")
    for directory in directories:
        if 'fig' in directory:
            print("\t" + directory)
            csvfile = directory + directory[len(parent_dir + 'fig_'):-1] + '.csv'
            t_i, xstar_i, ystar_i, zstar_i, x_i, y_i, z_i, vx_i, vy_i, vz_i, jpos_i, jvel_i, jtot_i, stdx_i, stdy_i = extract_data(csvfile)
            if not np.array(t_i == -np.inf).any():
                t.append(t_i)
                xstar.append(xstar_i)
                ystar.append(ystar_i)
                zstar.append(zstar_i)
                x.append(x_i)
                y.append(y_i)
                z.append(z_i)
                vx.append(vx_i)
                vy.append(vy_i)
                vz.append(vz_i)
                jpos.append(np.sqrt(jpos_i))
                jvel.append(jvel_i)
                jtot.append((jpos_i * 1e7 * (jvel_i + 1) + 1))
                stdx.append(stdx_i)
                stdy.append(stdy_i)
    return t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot, stdx, stdy

def get_data_depth_n(parent_dir, n):
    directories = get_dirs_depth_n(parent_dir, n)
    t = []
    xstar = []
    ystar = []
    zstar = []
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []
    jpos = []
    jvel = []
    jtot = []
    stdx = []
    stdy = []
    print("Extracting from...")
    for directory in directories:
        if 'fig' in directory:
            print("\t" + directory)
            csvfile = directory + directory[len(parent_dir + 'fig_'):-1] + '.csv'
            # csvfile = directory + directory[len('./mpcs/fig_'):-1] + '.csv'
            t_i, xstar_i, ystar_i, zstar_i, x_i, y_i, z_i, vx_i, vy_i, vz_i, jpos_i, jvel_i, jtot_i, stdx_i, stdy_i = extract_data(csvfile)
            if not np.array(t_i == -np.inf).any():
                t.append(t_i)
                xstar.append(xstar_i)
                ystar.append(ystar_i)
                zstar.append(zstar_i)
                x.append(x_i)
                y.append(y_i)
                z.append(z_i)
                vx.append(vx_i)
                vy.append(vy_i)
                vz.append(vz_i)
                jpos.append(np.sqrt(jpos_i))
                jvel.append(jvel_i)
                jtot.append((jpos_i * 1e7 * (jvel_i + 1) + 1))
                stdx.append(stdx_i)
                stdy.append(stdy_i)
    return t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot, stdx, stdy

def get_mean_and_var(t, x, m):
    t = t[:m]
    x = x[:m]
    N = 1000
    ti = np.zeros([len(x),N])
    xi = np.zeros([len(x),N])
    ti = np.linspace(0, 45*3600.0, N)
    for i in range(len(x)):
        if x[i].shape[0] == 0:
            print("WOAH what happened")
            print x[i]
        else:
            xi[i] = griddata(t[i], x[i], ti, method='cubic')
    mu = np.mean(xi, axis=0)
    sigma = np.std(xi, axis=0)
    return ti, mu, sigma

def get_lost_ones(t, x, m, t_neverlost):
    t = t[:m]
    x = x[:m]
    t_lost = []
    idx_notlost = np.zeros(m)
    t_lost_buffer = 3600 # in the simulation, if it was lost, it was really lost an hour before the simulation ended
    for i in range(len(x)):
        if t[i][-1] < t_neverlost:
            t_lost.append(t[i][-1] - t_lost_buffer)
        else:
            idx_notlost[i] = 1
    return np.array(t_lost), np.array(idx_notlost, dtype=bool)

def plot_n_samples(ax_jpos, ax_jvel, ax_jtot, parent_dir, m, n, c):
    t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot, stdx, stdy = get_data_n_samples(parent_dir, n)
    jpos_avg = []
    jvel_avg = []
    jtot_avg = []
    t_avg = []
    n_lost = 0
    for i in range(m):
        if np.max(t[i] > 45 * 3600):
            jpos_avg.append(np.cumsum(jpos[i]) / (t[i]+1))
            jvel_avg.append(np.cumsum(jvel[i]) / (t[i]+1))
            jtot_avg.append(np.cumsum(jtot[i]) / (t[i]+1))
            t_avg.append(t[i])
        else:
            n_lost += 1
    print(str(n_lost) + " for " + str(n) + "-sample system")
    t_jpos, mu_jpos, sigma_jpos = get_mean_and_var(t_avg, jpos_avg, m)
    t_jvel, mu_jvel, sigma_jvel = get_mean_and_var(t_avg, jvel_avg, m)
    t_jtot, mu_jtot, sigma_jtot = get_mean_and_var(t_avg, jtot_avg, m)
    print("Plotting...")
    # for i in range(40):
    ax_jpos.plot(t_jpos / 3600.0, mu_jpos / 1000.0, c=c)
    ax_jpos.plot(t_jpos / 3600.0, (mu_jpos+2*sigma_jpos) / 1000.0, linestyle='dashed', c=c)
    ax_jpos.plot(t_jpos / 3600.0, (mu_jpos-2*sigma_jpos) / 1000.0, linestyle='dashed', c=c)
    ax_jvel.plot(t_jvel / 3600.0, mu_jvel, c=c)
    ax_jvel.plot(t_jvel / 3600.0, mu_jvel+2*sigma_jvel, linestyle='dashed', c=c)
    ax_jvel.plot(t_jvel / 3600.0, mu_jvel-2*sigma_jvel, linestyle='dashed', c=c)
    ax_jtot.plot(t_jtot / 3600.0, mu_jtot / 1e16, c=c)
    ax_jtot.plot(t_jtot / 3600.0, (mu_jtot+2*sigma_jtot) / 1e16, linestyle='dashed', c=c)
    ax_jtot.plot(t_jtot / 3600.0, (mu_jtot-2*sigma_jtot) / 1e16, linestyle='dashed', c=c)
    # gray -= dgray
    ax_jpos.set_xlim([0,45])
    ax_jpos.set_ylim([0,20])
    ax_jvel.set_ylim([0,0.4])
    ax_jtot.set_ylim([0,3])
    return 1.0 * n_lost / m

def plot_depth_n(ax_jpos, ax_jvel, ax_jtot, parent_dir, m, n, c):
    t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot, stdx, stdy = get_data_depth_n(parent_dir, n)
    jpos_avg = []
    jvel_avg = []
    jtot_avg = []
    t_avg = []
    n_lost = 0
    for i in range(m):
        if np.max(t[i] > 45 * 3600):
            jpos_avg.append(np.cumsum(jpos[i]) / (t[i]+1))
            jvel_avg.append(np.cumsum(jvel[i]) / (t[i]+1))
            jtot_avg.append(np.cumsum(jtot[i]) / (t[i]+1))
            t_avg.append(t[i])
        else:
            n_lost += 1
    # print(str(n_lost) + " for " + str(n) + "-sample system")
    t_jpos, mu_jpos, sigma_jpos = get_mean_and_var(t_avg, jpos_avg, m)
    t_jvel, mu_jvel, sigma_jvel = get_mean_and_var(t_avg, jvel_avg, m)
    t_jtot, mu_jtot, sigma_jtot = get_mean_and_var(t_avg, jtot_avg, m)
    print("Plotting...")
    # for i in range(40):
    ax_jpos.plot(t_jpos / 3600.0, mu_jpos / 1000.0, c=c)
    ax_jpos.plot(t_jpos / 3600.0, (mu_jpos+2*sigma_jpos) / 1000.0, linestyle='dashed', c=c)
    ax_jpos.plot(t_jpos / 3600.0, (mu_jpos-2*sigma_jpos) / 1000.0, linestyle='dashed', c=c)
    ax_jvel.plot(t_jvel / 3600.0, mu_jvel, c=c)
    ax_jvel.plot(t_jvel / 3600.0, mu_jvel+2*sigma_jvel, linestyle='dashed', c=c)
    ax_jvel.plot(t_jvel / 3600.0, mu_jvel-2*sigma_jvel, linestyle='dashed', c=c)
    ax_jtot.plot(t_jtot / 3600.0, mu_jtot / 1e16, c=c)
    ax_jtot.plot(t_jtot / 3600.0, (mu_jtot+2*sigma_jtot) / 1e16, linestyle='dashed', c=c)
    ax_jtot.plot(t_jtot / 3600.0, (mu_jtot-2*sigma_jtot) / 1e16, linestyle='dashed', c=c)
    # gray -= dgray
    ax_jpos.set_xlim([0,45])
    ax_jpos.set_ylim([0,10])
    ax_jvel.set_ylim([0.15,0.25])
    ax_jtot.set_ylim([0,1])
    return 1.0 * n_lost / m

def plot_depth_n_all(ax_jpos, ax_jvel, ax_jtot, parent_dir, m, n, c):
    t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot, stdx, stdy = get_data_depth_n(parent_dir, n)
    jpos_avg = []
    jvel_avg = []
    jtot_avg = []
    t_avg = []
    n_lost = 0
    print("Plotting...")
    # for i in range(40):
    ax_std = ax_jpos.twinx()
    for i in range(len(jpos)):
        stdx_i = np.sqrt(np.cumsum(stdx[i]**2 * 5))
        stdy_i = np.sqrt(np.cumsum(stdy[i]**2 * 5))
        ax_jpos.plot(t[i] / 3600.0, jpos[i] / 1000.0, c=c)
        ax_jvel.plot(t[i] / 3600.0, jvel[i], c=c)
        ax_jtot.plot(t[i] / 3600.0, jtot[i] / 1e16, c=c)
        ax_std.plot(t[i] / 3600.0, stdx_i, c=c, linestyle='dashed')
        ax_std.plot(t[i] / 3600.0, stdy_i, c=c, linestyle='dotted')
    # gray -= dgray
    ax_jpos.set_xlim([0,45])
    ax_jpos.set_ylim([0,150])
    ax_jvel.set_ylim([0,2.1])
    ax_jtot.set_ylim([0,50])
    ax_std.set_ylim([0,3000])
    return 1.0 * n_lost / m

def plot1():
    fig = plt.figure()
    parent_dir = "./naives/"
    gs = gridspec.GridSpec(3,2)
    # ax_lost = fig.add_subplot(gs[0,0])
    ax_jpos_2 = fig.add_subplot(gs[0,0])
    ax_jvel_2 = fig.add_subplot(gs[1,0], sharex=ax_jpos_2)
    ax_jtot_2 = fig.add_subplot(gs[2,0], sharex=ax_jpos_2)
    ax_jpos_all = fig.add_subplot(gs[0,1])
    ax_jvel_all = fig.add_subplot(gs[1,1], sharex=ax_jpos_all)
    ax_jtot_all = fig.add_subplot(gs[2,1], sharex=ax_jpos_all)
    # ax_jpos_3 = fig.add_subplot(gs[9:16,1], sharey=ax_jpos_2)
    # ax_jvel_3 = fig.add_subplot(gs[17:24,1], sharex=ax_jpos_3, sharey=ax_jvel_2)
    # ax_jtot_3 = fig.add_subplot(gs[25:,1], sharex=ax_jpos_3, sharey=ax_jtot_2)
    # ax_jpos_4 = fig.add_subplot(gs[9:16,2], sharey=ax_jpos_2)
    # ax_jvel_4 = fig.add_subplot(gs[17:24,2], sharex=ax_jpos_4, sharey=ax_jvel_2)
    # ax_jtot_4 = fig.add_subplot(gs[25:,2], sharex=ax_jpos_4, sharey=ax_jtot_2)
    # ax_jpos_5 = fig.add_subplot(gs[9:16,3], sharey=ax_jpos_2)
    # ax_jvel_5 = fig.add_subplot(gs[17:24,3], sharex=ax_jpos_5, sharey=ax_jvel_2)
    # ax_jtot_5 = fig.add_subplot(gs[25:,3], sharex=ax_jpos_5, sharey=ax_jtot_2)
    # lost_2 = plot_n_samples(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, 70, 2, np.ones(3)*0.8)
    # lost_3 = plot_n_samples(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, 70, 4, np.ones(3)*0.6)
    # lost_4 = plot_n_samples(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, 70, 6, np.ones(3)*0.4)
    # lost_5 = plot_n_samples(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, 70, 8, np.ones(3)*0.2)
    parent_dir = "./mpcs/"
    plot_depth_n(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, 1, 1, np.ones(3)*0.8)
    plot_depth_n(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, 1, 2, np.ones(3)*0.6)
    plot_depth_n(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, 1, 3, np.ones(3)*0.4)
    plot_depth_n(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, 1, 4, np.ones(3)*0.2)
    plot_depth_n_all(ax_jpos_all, ax_jvel_all, ax_jtot_all, parent_dir, 1, 1, np.ones(3)*0.8)
    plot_depth_n_all(ax_jpos_all, ax_jvel_all, ax_jtot_all, parent_dir, 1, 2, np.ones(3)*0.6)
    plot_depth_n_all(ax_jpos_all, ax_jvel_all, ax_jtot_all, parent_dir, 1, 3, np.ones(3)*0.4)
    plot_depth_n_all(ax_jpos_all, ax_jvel_all, ax_jtot_all, parent_dir, 1, 4, np.ones(3)*0.2)
    # n_samples = np.array([2,4,6,8])
    # lost = np.array([lost_2, lost_3, lost_4, lost_5])
    # colors = np.array([np.ones(3)*0.8, np.ones(3)*0.6, np.ones(3)*0.4, np.ones(3)*0.2])
    # ax_lost.bar(n_samples, lost, color=colors)
    # ax_lost.set_xticks(n_samples)
    # ax_lost.set_ylim([0,1])
    # ax_lost.set_xlabel("Number of Candidate Altitudes")
    ax_jtot_2.set_xlabel("Simulation Time [hr]")
    # ax_lost.set_ylabel("Proportion of\nAgents Lost")
    ax_jpos_2.set_ylabel("Mean Distance\nfrom Set Point [km]")
    ax_jvel_2.set_ylabel("Mean Direction\nQuality")
    ax_jtot_2.set_ylabel("Planning Cost")
    ax_jpos_all.set_ylabel("Distance from\nSet Point [km]")
    ax_jvel_all.set_ylabel("Direction Quality")
    ax_jtot_all.set_ylabel("Planning Cost")
    ax_jtot_all.set_xlabel("Simulation Time [hr]")
    plt.setp(ax_jpos_2.get_xticklabels(), visible=False)
    plt.setp(ax_jvel_2.get_xticklabels(), visible=False)
    plt.setp(ax_jpos_all.get_xticklabels(), visible=False)
    plt.setp(ax_jvel_all.get_xticklabels(), visible=False)
    # plt.tight_layout()
    plt.show()

def plot2():
    fig = plt.figure()
    gs = gridspec.GridSpec(3,2)

    ax_jpos_2 = fig.add_subplot(gs[0,0])
    ax_jvel_2 = fig.add_subplot(gs[1,0], sharex=ax_jpos_2)
    ax_jtot_2 = fig.add_subplot(gs[2,0], sharex=ax_jpos_2)

    ax_jpos_all = fig.add_subplot(gs[0,1])
    ax_jvel_all = fig.add_subplot(gs[1,1], sharex=ax_jpos_all)
    ax_jtot_all = fig.add_subplot(gs[2,1], sharex=ax_jpos_all)

    parent_dir = "./quadcosts/y/"
    plot_depth_n(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, 4, 4, np.ones(3)*0.8)
    plot_depth_n_all(ax_jpos_all, ax_jvel_all, ax_jtot_all, parent_dir, 4, 4, np.ones(3)*0.8)
    parent_dir = "./quadcosts/n/"
    plot_depth_n(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, 4, 4, np.ones(3)*0.2)
    plot_depth_n_all(ax_jpos_all, ax_jvel_all, ax_jtot_all, parent_dir, 4, 4, np.ones(3)*0.2)


    ax_jtot_2.set_xlabel("Simulation Time [hr]")
    ax_jpos_2.set_ylabel("Mean Distance\nfrom Set Point [km]")
    ax_jvel_2.set_ylabel("Mean Direction\nQuality")
    ax_jtot_2.set_ylabel("Planning Cost")
    ax_jpos_all.set_ylabel("Distance from\nSet Point [km]")
    ax_jvel_all.set_ylabel("Direction Quality")
    ax_jtot_all.set_ylabel("Planning Cost")
    ax_jtot_all.set_xlabel("Simulation Time [hr]")
    plt.setp(ax_jpos_2.get_xticklabels(), visible=False)
    plt.setp(ax_jvel_2.get_xticklabels(), visible=False)
    plt.setp(ax_jpos_all.get_xticklabels(), visible=False)
    plt.setp(ax_jvel_all.get_xticklabels(), visible=False)
    plt.show()

def plot3():
    fig = plt.figure()
    gs = gridspec.GridSpec(3,2)

    ax_jpos_2 = fig.add_subplot(gs[0,0])
    ax_jvel_2 = fig.add_subplot(gs[1,0], sharex=ax_jpos_2)
    ax_jtot_2 = fig.add_subplot(gs[2,0], sharex=ax_jpos_2)

    ax_jpos_all = fig.add_subplot(gs[0,1])
    ax_jvel_all = fig.add_subplot(gs[1,1], sharex=ax_jpos_all)
    ax_jtot_all = fig.add_subplot(gs[2,1], sharex=ax_jpos_all)

    m = 3
    parent_dir = "./quadcosts/sampling_scenarios/w-sampling/05-min/"
    plot_depth_n(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, m, 4, np.ones(3)*0.8)
    plot_depth_n_all(ax_jpos_all, ax_jvel_all, ax_jtot_all, parent_dir, m, 4, np.ones(3)*0.8)
    parent_dir = "./quadcosts/sampling_scenarios/w-sampling/10-min/"
    plot_depth_n(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, m, 4, np.ones(3)*0.5)
    plot_depth_n_all(ax_jpos_all, ax_jvel_all, ax_jtot_all, parent_dir, m, 4, np.ones(3)*0.5)
    parent_dir = "./quadcosts/sampling_scenarios/w-sampling/20-min/"
    plot_depth_n(ax_jpos_2, ax_jvel_2, ax_jtot_2, parent_dir, m, 4, np.ones(3)*0.2)
    plot_depth_n_all(ax_jpos_all, ax_jvel_all, ax_jtot_all, parent_dir, m, 4, np.ones(3)*0.2)


    ax_jtot_2.set_xlabel("Simulation Time [hr]")
    ax_jpos_2.set_ylabel("Mean Distance\nfrom Set Point [km]")
    ax_jvel_2.set_ylabel("Mean Direction\nQuality")
    ax_jtot_2.set_ylabel("Planning Cost")
    ax_jpos_all.set_ylabel("Distance from\nSet Point [km]")
    ax_jvel_all.set_ylabel("Direction Quality")
    ax_jtot_all.set_ylabel("Planning Cost")
    ax_jtot_all.set_xlabel("Simulation Time [hr]")
    plt.setp(ax_jpos_2.get_xticklabels(), visible=False)
    plt.setp(ax_jvel_2.get_xticklabels(), visible=False)
    plt.setp(ax_jpos_all.get_xticklabels(), visible=False)
    plt.setp(ax_jvel_all.get_xticklabels(), visible=False)
    plt.show()

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

def running_avg(d):
    avg = np.cumsum(d) / np.linspace(1, len(d), len(d))
    return avg

def running_avg_and_std(d):
    avg_d = np.zeros(len(d))
    std_d = np.zeros(len(d))
    for i in range(len(d)):
        if i > 0:
            avg_d[i] = np.mean(d[:i])
            std_d[i] = np.std(d[:i])
    return avg_d, std_d

def plot_distribution_of_mean(ax, data, idx, **kwargs):
    scale = pyutils.parsekw(kwargs, 'scale', 1.)
    tidx = pyutils.parsekw(kwargs, 'tidx', 0)
    run_avg = pyutils.parsekw(kwargs, 'run_avg', True)
    c = pyutils.parsekw(kwargs, 'c', 'k')
    di = []
    ti = []
    for d in data:
        d_float = np.array(d[idx], dtype=float)
        t_float = np.array(d[tidx], dtype=float)
        if run_avg:
            di.append(running_avg(d_float))
        else:
            di.append(d_float)
        ti.append(t_float)
    ti = np.array(ti)
    m = 25
    t, mu, sigma = get_mean_and_var(ti, di, m)
    t = t / 3600. # sec to hr
    mu = mu * scale
    sigma = sigma * scale
    ax.plot(t, mu, c=c)
    ax.plot(t, mu+2*sigma, c=c, linestyle='dashed')
    ax.plot(t, mu-2*sigma, c=c, linestyle='dashed')

def plot_running_avg_and_std(ax, data, idx, **kwargs):
    scale = pyutils.parsekw(kwargs, 'scale', 1.)
    tidx = pyutils.parsekw(kwargs, 'tidx', 0)
    run_avg = pyutils.parsekw(kwargs, 'run_avg', True)
    c = pyutils.parsekw(kwargs, 'c', 'k')
    uncertainties = pyutils.parsekw(kwargs, 'uncertainties', False)
    ax_dist = pyutils.parsekw(kwargs, 'ax_dist', ax)
    ax_var = pyutils.parsekw(kwargs, 'ax_var', ax)
    linestyle = pyutils.parsekw(kwargs, 'linestyle', '-')
    di = []
    di_std = []
    ti = []
    for d in data:
        d_float = np.array(d[idx], dtype=float)
        t_float = np.array(d[tidx], dtype=float)
        mu, sigma = running_avg_and_std(d_float)
        di.append(mu)
        di_std.append(sigma)
        ti.append(t_float)
    ti = np.array(ti)
    m = len(di)
    lost_ones, idx_notlost = get_lost_ones(ti, di, m, 46*3600)
    p_lost = 1. - 1.*np.sum(idx_notlost) / len(idx_notlost)
    ti_notlost = []
    di_notlost = []
    di_std_notlost = []
    for i in range(len(di)):
        if idx_notlost[i]:
            ti_notlost.append(ti[i])
            di_notlost.append(di[i])
            di_std_notlost.append(di_std[i])
    di_t, di_mu, di_sigma = get_mean_and_var(ti_notlost, di_notlost, m)
    di_std_t, di_std_mu, di_std_sigma = get_mean_and_var(ti_notlost, di_std_notlost, m)
    di_t = di_t / 3600. # sec to hr
    di_mu = di_mu * scale
    di_std_mu = di_std_mu * scale
    di_sigma = di_sigma * scale
    di_std_sigma = di_std_sigma * scale
    di_mu_uncertainty = 2 * di_sigma
    di_std_uncertainty = di_mu_uncertainty + 2 * di_std_sigma
    if len(lost_ones) == 0:
        lost_cdf = np.zeros(len(di_t))
    else:
        lost_mu = np.mean(lost_ones)
        lost_sigma = np.std(lost_ones)
        lost_cdf = p_lost * norm.cdf(di_t*3600, lost_mu, lost_sigma)
    if ax == ax_var:
        h, = ax.plot(di_t, di_mu, c=c)
        ax.plot(di_t, di_mu+2*di_std_mu, c=c*0.7, linestyle='dashed')
        ax.plot(di_t, di_mu-2*di_std_mu, c=c*0.7, linestyle='dashed')
    else:
        h, = ax.plot(di_t, di_mu, c=c, linestyle=linestyle)
        ax_var.plot(di_t, di_std_mu, c=c, linestyle=linestyle)
    ax_dist.plot(di_t, lost_cdf, c=c, linestyle=linestyle)
    if uncertainties:
        ax.plot(di_t, di_mu+di_mu_uncertainty, c=c, linestyle='dotted')
        ax.plot(di_t, di_mu-di_mu_uncertainty, c=c, linestyle='dotted')
        ax.plot(di_t, di_mu+2*di_std_mu+di_std_uncertainty, c=c*0.7, linestyle='dotted')
        ax.plot(di_t, di_mu+2*di_std_mu-di_std_uncertainty, c=c*0.7, linestyle='dotted')
        ax.plot(di_t, di_mu-2*di_std_mu+di_std_uncertainty, c=c*0.5, linestyle='dotted')
        ax.plot(di_t, di_mu-2*di_std_mu-di_std_uncertainty, c=c*0.5, linestyle='dotted')
    return h

def plot_all(ax, data, idx, **kwargs):
    scale = pyutils.parsekw(kwargs, 'scale', 1.)
    tidx = pyutils.parsekw(kwargs, 'tidx', 0)
    run_avg = pyutils.parsekw(kwargs, 'run_avg', True)
    c = pyutils.parsekw(kwargs, 'c', 'k')
    linestyle = pyutils.parsekw(kwargs, 'linestyle', '-')
    max_gray = 0.4
    min_gray = 1.
    del_gray = abs(max_gray - min_gray) / len(data)
    gray = min_gray
    for d in data:
        d_float = np.array(d[idx], dtype=float)
        t_float = np.array(d[tidx], dtype=float)
        if run_avg:
            di = running_avg(d_float) * scale
        else:
            di = d_float * scale
        ti = t_float / 3600. # sec to hr
        ax.plot(ti, di, c=c, alpha=gray, linestyle=linestyle)
        gray -= del_gray

def process4(data, m):
    processed = []
    i = 0
    for d in data:
        t = []
        x = []
        y = []
        vx = []
        vy = []
        jp = []
        jv = []
        jt = []
        for row in d:
            ti = row[0]
            xi = row[1]
            yi = row[2]
            vxi = row[5][1:-1]
            vyi = row[6][1:-1]
            jpi = Jp(p=np.array([xi, yi], dtype=float), pstar=np.zeros(2))
            jvi = Jv(p=np.array([xi, yi], dtype=float), pstar=np.zeros(2), pdot=np.array([vxi, vyi], dtype=float))
            jti = jpi * 1e0 + jvi * 1e5
            t.append(ti)
            x.append(xi)
            y.append(yi)
            vx.append(vxi)
            vy.append(vyi)
            jp.append(jpi)
            jv.append(jvi)
            jt.append(jti)
        processed_i = [t, x, y, vx, vy, jp, jv, jt]
        processed.append(processed_i)
        i += 1
        print(str(i) + "/" + str(m))
        if i >= m:
            break
    return processed

def process5(data, m):
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

def plot4_helper(ax_jp, ax_jv, ax_jt, ax_jp_all, ax_jv_all, ax_jt_all, parent_dir, case_no, depth, c):
    path = parent_dir \
        + "/case-" + str(case_no) \
        + "/depth-" + str(depth) + "/"
    print path
    directories = glob(path + "*/")
    raw_data = []
    for d in directories:
        print d
        f = d + d[len(path + 'fig_'):-1] + '.csv'
        raw_data.append(extract_data_no_fields(f))
    m = len(directories)
    print "processing"
    data = process4(raw_data, m)

    print "plotting 1"
    plot_running_avg_and_std(ax_jp, data, 5, scale=1e-3, run_avg=False, c=c)
    plot_running_avg_and_std(ax_jv, data, 6, scale=1e-3, run_avg=False, c=c)
    plot_running_avg_and_std(ax_jt, data, 7, scale=1e-3, run_avg=False, c=c)

    print "plotting 2"
    plot_all(ax_jp_all, data, 5, scale=1e-3, run_avg=False, c=c)
    plot_all(ax_jv_all, data, 6, scale=1e-3, run_avg=False, c=c)
    plot_all(ax_jt_all, data, 7, scale=1e-3, run_avg=False, c=c)

def plot5_helper(ax_jp, ax_jp_var, ax_jp_all, ax_jp_dist,  parent_dir, case_no, depth, c, linestyle):
    path = parent_dir \
        + "/case-" + str(case_no) \
        + "/depth-" + str(depth) + "/"
    print path
    directories = glob(path + "*/")
    raw_data = []
    for d in directories:
        print d
        f = d + d[len(path + 'fig_'):-1] + '.csv'
        raw_data.append(extract_data_no_fields(f))
    m = len(directories)
    print "processing"
    data = process4(raw_data, m)

    print "plotting 1"
    h = plot_running_avg_and_std(ax_jp, data, 5, scale=1e-3, run_avg=False, c=c, linestyle=linestyle, ax_dist=ax_jp_dist, ax_var=ax_jp_var)

    print "plotting 2"
    plot_all(ax_jp_all, data, 5, scale=1e-3, run_avg=False, c=c, linestyle=linestyle)

    return h

def plot4():
    fig = plt.figure()
    gs = gridspec.GridSpec(3,2)

    ax_jp = fig.add_subplot(gs[0,0])
    ax_jv = fig.add_subplot(gs[1,0], sharex=ax_jp)
    ax_jt = fig.add_subplot(gs[2,0], sharex=ax_jp)

    ax_jp_all = fig.add_subplot(gs[0,1])
    ax_jv_all = fig.add_subplot(gs[1,1], sharex=ax_jp_all)
    ax_jt_all = fig.add_subplot(gs[2,1], sharex=ax_jp_all)

    parent_dir = "./datasets"
    case_no = '4a'
    depth = 4
    c = np.array([0.,0.,1.])
    plot4_helper(ax_jp, ax_jv, ax_jt, ax_jp_all, ax_jv_all, ax_jt_all, parent_dir, case_no, depth, c)

    parent_dir = "./datasets"
    case_no = '2a'
    depth = 4
    c = np.array([0.,1.,0.])
    plot4_helper(ax_jp, ax_jv, ax_jt, ax_jp_all, ax_jv_all, ax_jt_all, parent_dir, case_no, depth, c)

    plt.show()

def plot5(cases, depths, labels):
    fig = plt.figure()
    gs = gridspec.GridSpec(4,2)

    ax_jp = fig.add_subplot(gs[0,:])
    ax_jp_var = fig.add_subplot(gs[1,:], sharex=ax_jp)
    ax_jp_dist = fig.add_subplot(gs[2,:], sharex=ax_jp)
    ax_jp_all = fig.add_subplot(gs[3,:], sharex=ax_jp)

    # n = 5
    # my_sm = sm(cmap='jet')
    # my_sm.set_clim([0,n])
    # c = my_sm.to_rgba(np.linspace(1,n,n))[:,0:3]
    c = [np.array([0., 0., 0.]), \
        np.array([0., 0., 1.]), \
        np.array([0., 1., 0.]), \
        np.array([1., 0., 0.]), \
        np.array([1., 1., 0.]), \
        np.array([1., 0., 1.]), \
        np.array([0., 1., 1.]), \
        np.array([1., 1., 1.])]

    linestyle = ['solid', \
                'dashed', \
                'dotted' ]

    parent_dir = "./datasets"
    h = []
    for i in range(len(cases)):
        case_no = cases[i]
        depth = depths[i]
        color = c[i%len(c)]
        line = linestyle[np.int(np.floor(1.*i/len(c)))]
        # print color
        hi = plot5_helper(ax_jp, ax_jp_var, ax_jp_all, ax_jp_dist, parent_dir, case_no, depth, color, line)
        h.append(hi)

    plt.legend(h, \
            labels, \
            loc='center right', \
            bbox_to_anchor=(1.,0.5),
            bbox_transform=plt.gcf().transFigure)

    axes = [ax_jp, ax_jp_var, ax_jp_dist, ax_jp_all]
    gray = 0.7
    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.set_facecolor(gray*np.ones(3))
        ax.grid(visible=True, c=gray*np.ones(3)/2)

    plt.setp(ax_jp.get_xticklabels(), visible=False)
    plt.setp(ax_jp_var.get_xticklabels(), visible=False)
    plt.setp(ax_jp_dist.get_xticklabels(), visible=False)

    ax_jp.set_xlim([0,45])

    ax_jp.set_ylim([0,100])
    ax_jp_var.set_ylim([0,50])
    ax_jp_dist.set_ylim([0,1])
    ax_jp_all.set_ylim([0,125])

    ax_jp_all.set_xlabel('Time [hrs]')

    ax_jp.set_ylabel("Mean Distance [km]")
    ax_jp_var.set_ylabel("Std of Distance [km]")
    ax_jp_dist.set_ylabel("P(lost)")
    ax_jp_all.set_ylabel("Distance [km]")

    ax_jp.set_title("Planner Performance")

    plt.show()

def plot6_helper(axes, parent_dir, case_no, depth, c, linestyle, n):
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
    data = process5(raw_data, m)

    T = 45.*3600
    N = 1000
    t_last = 0.
    idx_last = 0
    t_window = T / n
    tidx = 0
    jidx = 5
    idx = 0
    data = pyutils.grid_uneven_data(data, 0, T, N)
    t = data[0][tidx]
    mu_x = []
    mu_y = []
    mu_jp = []
    sigma_x = []
    sigma_y = []
    sigma_jp = []
    times = []
    for i in range(n):
        while t[idx] - t_last < t_window and idx < N - 1:
            idx += 1
        jp = []
        x = []
        y = []
        for d in data:
            jpi = d[jidx]
            xi = -d[1]*1e-3
            yi = d[2]*1e-3
            jpi[np.isnan(jpi)] = 0.
            xi[np.isnan(xi)] = 0.
            yi[np.isnan(yi)] = 0.
            jp.extend(jpi[idx_last:idx])
            x.extend(xi[idx_last:idx])
            y.extend(yi[idx_last:idx])
        cov = np.cov(x, y)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        mu_x.append(np.mean(x))
        mu_y.append(np.mean(y))
        mu_jp.append(np.mean(jp))
        sigma_x.append(np.std(x))
        sigma_y.append(np.std(y))
        sigma_jp.append(np.std(jp))
        times.append(t[idx] / 3600)
        r_mean = mu_jp[-1] * 1e-3
        r_min = np.max([0., mu_jp[-1] - 2*sigma_jp[-1]]) * 1e-3
        r_max = (mu_jp[-1] + 2*sigma_jp[-1]) * 1e-3
        # c_mean = Circle([0,0], r_mean, facecolor='none', edgecolor=c, linestyle='solid')
        # c_min = Circle([0,0], r_min, facecolor='none', edgecolor=c, linestyle='dashed')
        # c_max = Circle([0,0], r_max, facecolor='none', edgecolor=c, linestyle='dashed')
        ell = Ellipse(xy=(mu_x[-1], mu_y[-1]), width=lambda_[0]*4, height=lambda_[1]*4, angle=np.rad2deg(np.arccos(v[0,0])), facecolor='none', edgecolor=c, linestyle='solid')
        # ell = Ellipse([mu_x[-1], mu_y[-1]], 4*sigma_x[-1], 4*sigma_y[-1], facecolor='none', edgecolor=c, linestyle='solid')
        # patches = [c_mean, c_min, c_max]
        # patches_mean = [c_mean]
        # patches_bounds = [c_max]
        patches_ell = [ell]
        # collection_mean = PatchCollection(patches_mean, facecolor='none', edgecolor=c, linestyle='solid')
        # collection_bounds = PatchCollection(patches_bounds, facecolor='none', edgecolor=c, linestyle='dashed')
        collection_ell = PatchCollection(patches_ell, facecolor=c, edgecolor='none', linestyle='solid', alpha=0.2)
        axes[i].scatter(mu_x[-1], mu_y[-1], c=c, s=10)
        # axes[i].add_collection(collection_mean)
        # axes[i].add_collection(collection_bounds)
        axes[i].add_collection(collection_ell)
        axes[i].set_xlim([-100,100])
        axes[i].set_ylim([-100,100])
        t_last = t[idx]
        idx_last = idx

    T = 45.*3600
    N = 1000
    t_last = 0.
    tidx = 0
    jidx = 5
    mu_x = []
    mu_y = []
    mu_jp = []
    sigma_x = []
    sigma_y = []
    sigma_jp = []
    times = []
    jp_all = []
    for i in range(N):
        jp = []
        x = []
        y = []
        for d in data:
            jpi = d[jidx]*1e-3
            xi = -d[1]*1e-3
            yi = d[2]*1e-3
            jpi[np.isnan(jpi)] = 0.
            xi[np.isnan(xi)] = 0.
            yi[np.isnan(yi)] = 0.
            jp.append(jpi[i])
            x.append(xi[i])
            y.append(yi[i])
            jp_all.append(jpi[i])
        mu_x.append(np.mean(x))
        mu_y.append(np.mean(y))
        mu_jp.append(np.mean(jp))
        sigma_x.append(np.std(x))
        sigma_y.append(np.std(y))
        sigma_jp.append(np.std(jp))
        times.append(t[i] / 3600)
    print('mean:\t' + str(np.mean(jp_all)))
    print('std:\t' + str(np.std(jp_all)))
    print('5th:\t' + str(np.percentile(jp_all, 5.)))
    print('25th:\t' + str(np.percentile(jp_all, 25.)))
    print('75th:\t' + str(np.percentile(jp_all, 75.)))
    print('95th:\t' + str(np.percentile(jp_all, 95.)))
    # plt.figure()
    # plt.hist(jp_all,100)
    # plt.show()

    # axes[0].legend(labels)
    return times, mu_x, mu_y, mu_jp, sigma_x, sigma_y, sigma_jp, jp_all

def plot6(cases, depths, labels, n):
    fig = plt.figure()
    fig.set_size_inches((10,5), forward=True)
    axes = []
    ax_trends_mu = []
    ax_trends_sigma = []
    im = plt.imread('stanford_area.png') # color
    # n = 12
    pics_per_row = 6
    if n % pics_per_row == 0:
        rows = n / pics_per_row
        gs = gridspec.GridSpec(2 * rows, pics_per_row)
        for i in range(rows):
            for j in range(pics_per_row):
                axes.append(fig.add_subplot(gs[2*i+1,j:(j+1)], aspect='equal'))
                axes[-1].imshow(im, extent=(-100,100,-100,100))
                if j == 0:
                    axes[-1].set_xlabel("x error, km")
                    axes[-1].set_ylabel("y error, km")
                else:
                    plt.setp(axes[-1].get_xticklabels(), visible=False)
                    plt.setp(axes[-1].get_yticklabels(), visible=False)
            ax_trends_mu.append(fig.add_subplot(gs[2*i,:]))
            ax_trends_mu[-1].set_xlabel("time, hr")
            ax_trends_mu[-1].set_ylabel("position error, km")
            # ax_trends_sigma.append(ax_trends_mu[-1].twinx())
    else:
        gs = gridspec.GridSpec(2, n)
        for i in range(n):
            axes.append(fig.add_subplot(gs[0,i:(i+1)], aspect='equal'))
            axes[i].imshow(im, extent=(-100,100,-100,100))
        ax_trends_mu.append(fig.add_subplot(gs[1,:]))
        ax_trends_sigma.append(ax_trends_mu[0].twinx())

    c = [np.array([0.2, 0.1, 0.2]), \
        np.array([0., 0., 1.]), \
        np.array([1., 0., 0.]), \
        np.array([0.9, 0., 0.9]), \
        np.array([0., 0.7, 0.]), \
        np.array([1., 1., 0.]), \
        np.array([1., 0., 1.]), \
        np.array([0., 1., 1.]), \
        np.array([1., 1., 1.])]

    linestyle = ['solid', \
                'dashed', \
                'dotted' ]

    parent_dir = "./datasets"
    histfig, histax = plt.subplots(len(cases), 1, sharex=True, sharey=True)
    histax[-1].set_xlabel('distance from setpoint, km')
    for i in range(len(cases)):
        print i
        case_no = cases[i]
        depth = depths[i]
        color = c[i%len(c)]
        line = linestyle[np.int(np.floor(1.*i/len(c)))]
        times, mu_x, mu_y, mu_jp, sigma_x, sigma_y, sigma_jp, jp_all = plot6_helper(axes, parent_dir, case_no, depth, color, line, n)
        histax[i].hist(jp_all, 100, color=color, density=True)
        histax[i].set_ylabel(labels[i])
        tplot = np.array(times)
        # tprev = 0.
        # for k in range(len(tplot)):
        #     tcurr = tplot[k]
        #     tplot[k] = (tcurr + tprev) / 2.
        #     tprev = tcurr
        if n % pics_per_row == 0:
            idx = 0
            idx_last = 0
            tlast = 0.
            t_window = 45. / rows
            __mu_all = []
            __sigma_all = []
            for j in range(rows):
                while tplot[idx] - tlast < t_window:
                    idx += 1
                    if idx >= 1000:
                        idx -= 1
                        break
                __mu = np.array(mu_jp[idx_last:idx])
                __t = tplot[idx_last:idx]
                __sigma = np.array(sigma_jp[idx_last:idx])
                __mu_all.extend(__mu)
                __sigma_all.extend(__sigma)
                ax_trends_mu[j].plot(__t, __mu, c=color)
                if j == 0:
                    ax_trends_mu[j].legend(labels, loc='upper left')
                ax_trends_mu[j].fill_between(__t, __mu+2*__sigma, __mu-2*__sigma, alpha=0.2, edgecolor='none', facecolor=color)
                ax_trends_mu[j].vlines(np.linspace(45.*j/rows, 45.*(j+1)/rows, pics_per_row+1), 0, 100, linestyles='dotted')
                # ax_trends_sigma[i].plot(tplot[idx_last:idx],np.array(sigma_jp[idx_last:idx]), c=color, linestyle='dashed')
                ax_trends_mu[j].set_xlim([45.*j/rows,45.*(j+1)/rows])
                ax_trends_mu[j].set_ylim([0,100])
                # ax_trends_sigma[i].set_ylim([0,0])
                idx_last = idx
                tlast = tplot[idx]
        else:
            ax_trends_mu[0].plot(tplot, np.array(mu_jp), c=color)
            ax_trends_sigma[0].plot(tplot, np.array(sigma_jp), c=color, linestyle='dashed')
            ax_trends_mu[0].set_xlim([0,45])
            ax_trends_mu[0].set_ylim([0,80])
            ax_trends_sigma[0].set_ylim([0,80])
        pyutils.phead(labels[i], 24)
        pyutils.pfields('', ['mean', 'std', 'min', 'max'])
        pyutils.prow('jp', [np.mean(__mu_all), np.std(__mu_all), np.min(__mu_all[20:]), np.max(__mu_all)])
        pyutils.prow('std',[np.mean(__sigma_all), np.std(__sigma_all), np.min(__sigma_all[20:]), np.max(__sigma_all)])
    # plt.tight_layout()
    plt.show()

def plot7_helper(ax, parent_dir, case_no, depth, c, linestyle, n):
    path = parent_dir \
        + "/case-" + str(case_no) \
        + "/depth-" + str(depth) + "/"
    print path
    directories = glob(path + "*/")
    raw_data = []
    for d in directories:
        print d
        f = d + d[len(path + 'fig_'):-1] + '.csv'
        raw_data.append(extract_data_no_fields(f))
    m = len(directories)
    print "processing"
    data = process5(raw_data, m)

    T = 45.*3600
    N = 1000
    data = pyutils.grid_uneven_data(data, 0, T, N)
    print(len(data))

    tidx = 0
    jidx = 5
    t_lost = []
    not_lost = 0
    for d in data:
        t = np.array(d[tidx])
        jp = np.array(d[jidx])
        t_lost_i = t[jp < jp[-1]][-1]
        t_lost.append(t_lost_i)
        if len(np.array(t_lost_i).shape) == 1:
            not_lost += 1
    lost_so_far = 0
    remaining = np.ones(len(data[0][tidx])) * m
    for i in range(len(t)):
        lost_so_far += np.sum(abs(t_lost - t[i]) < 1e-3)
        remaining[i] -= lost_so_far
    remaining = 100. * remaining / m
    ax.plot(t/3600, 100-remaining, c=c)
    ax.set_xlim([0,45])
    ax.set_ylim([0,100])
    ax.set_xlabel('time, hrs')
    ax.set_ylabel('agents lost, %')

def plot7(cases, depths, labels, n):
    fig = plt.figure()
    fig.set_size_inches((8,4), forward=True)
    ax = plt.gca()
    im = plt.imread('stanford_area.png') # color

    c = [np.array([0., 0.9, 0.9]), \
        np.array([0., 0., 0.]), \
        np.array([0.6, 0.6, 0.6]), \
        np.array([1., 0., 0.]), \
        np.array([0., 0., 1.]), \
        np.array([1., 0., 1.]), \
        np.array([0., 1., 1.]), \
        np.array([1., 1., 1.])]

    linestyle = ['solid', \
                'dashed', \
                'dotted' ]

    parent_dir = "./datasets"
    for i in range(len(cases)):
        case_no = cases[i]
        depth = depths[i]
        color = c[i%len(c)]
        line = linestyle[np.int(np.floor(1.*i/len(c)))]
        plot7_helper(ax, parent_dir, case_no, depth, color, line, n)

    ax.legend(labels)
    plt.show()

cases = ['2a', '2a', '2a-bad-pruning', '2a-bad-pruning']
depths = [1, 2, 1, 2]
labels = ['good 1', 'good 2', 'bad 1', 'bad 2']
# plot5(cases, depths, labels)

# cases = ['2a', '2a', '2b', '2b', '1a', '3a', '2m', '2m']
# depths = [2, 4, 2, 4, 1, 4, 2, 4]
cases = ['3a', '2a', '2b']
cases = ['3a', '2a', '2a']
# cases = ['3a', '1a']
cases = ['5a', '4a', '3a', '1a', '2a', '2a', '2b', '2b']
depths = [4, 4, 4]
depths = [4, 4, 2]
# depths = [4, 1]
depths = [4, 4, 4, 1, 4, 2, 4, 2]
labels = ['Myopic velocity selection', 'Tree search, depth 4', 'Deprived tree search, depth 4']
labels = ['Myopic velocity selection', 'Tree search, depth 4', 'Tree search, depth 2']
# labels = ['Myopic velocity selection', 'Myopic stream selection']
labels = ['Naive', 'Reactive', 'Myopic velocity selection', 'Myopic stream selection', 'Tree search, depth 4', 'Tree search, depth 2', 'Deprived tree search, depth 2', 'Deprived tree search, depth 4']
# labels = ['Tree search, depth 2', 'Deprived tree search, depth 2', 'Deprived tree search, depth 4']
# plot6(cases, depths, labels, 6)

cases = ['3a', '5a', '4a', '2a', '2b', '2b']
cases = ['5a', '4a', '3a', '1a', '2a', '2a', '2b', '2b']
depths = [4, 4, 4, 2, 2, 4]
depths = [4, 4, 4, 1, 4, 2, 4, 2]
labels = ['Myopic velocity selection', 'Naive', 'Reactive', 'Tree search, depth 2', 'Deprived tree search, depth 2', 'Deprived tree search, depth 4']
labels = ['Naive', 'Reactive', 'Myopic velocity selection', 'Myopic stream selection', 'Tree search, depth 4', 'Tree search, depth 2', 'Deprived tree search, depth 2', 'Deprived tree search, depth 4']
plot7(cases, depths, labels, 10)
