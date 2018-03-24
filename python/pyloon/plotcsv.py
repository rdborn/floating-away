import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib import gridspec
from scipy.interpolate import griddata
import os.path
import sys

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
                jtot.append(process(row[12]))
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
    return t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot

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
    print("Extracting from...")
    for directory in directories:
        print("\t" + directory)
        csvfile = directory + directory[len('./naives/fig_'):-1] + '.csv'
        t_i, xstar_i, ystar_i, zstar_i, x_i, y_i, z_i, vx_i, vy_i, vz_i, jpos_i, jvel_i, jtot_i = extract_data(csvfile)
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
    return t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot

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
    print("Extracting from...")
    for directory in directories:
        print("\t" + directory)
        csvfile = directory + directory[len('./mpcs/fig_'):-1] + '.csv'
        t_i, xstar_i, ystar_i, zstar_i, x_i, y_i, z_i, vx_i, vy_i, vz_i, jpos_i, jvel_i, jtot_i = extract_data(csvfile)
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
    return t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot

def get_mean_and_var(t, x, m):
    t = t[:m]
    x = x[:m]
    N = 100
    ti = np.zeros([len(x),N])
    xi = np.zeros([len(x),N])
    ti = np.linspace(0, 45*3600.0, N)
    for i in range(len(x)):
        xi[i] = griddata(t[i], x[i], ti, method='cubic')
    mu = np.mean(xi, axis=0)
    sigma = np.std(xi, axis=0)
    return ti, mu, sigma

def plot_n_samples(ax_jpos, ax_jvel, ax_jtot, parent_dir, m, n, c):
    t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot = get_data_n_samples(parent_dir, n)
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
    t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot = get_data_depth_n(parent_dir, n)
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
    print("Plotting...")
    # for i in range(40):
    ax_jpos.plot(t[0] / 3600.0, jpos_avg[0] / 1000.0, c=c)
    ax_jvel.plot(t[0] / 3600.0, jvel_avg[0], c=c)
    ax_jtot.plot(t[0] / 3600.0, jtot_avg[0] / 1e16, c=c)
    # gray -= dgray
    ax_jpos.set_xlim([0,45])
    ax_jpos.set_ylim([0,15])
    ax_jvel.set_ylim([0.1,0.4])
    ax_jtot.set_ylim([0,2])
    return 1.0 * n_lost / m

def plot_depth_n_all(ax_jpos, ax_jvel, ax_jtot, parent_dir, m, n, c):
    t, xstar, ystar, zstar, x, y, z, vx, vy, vz, jpos, jvel, jtot = get_data_depth_n(parent_dir, n)
    jpos_avg = []
    jvel_avg = []
    jtot_avg = []
    t_avg = []
    n_lost = 0
    print("Plotting...")
    # for i in range(40):
    ax_jpos.plot(t[0] / 3600.0, jpos[0] / 1000.0, c=c)
    ax_jvel.plot(t[0] / 3600.0, jvel[0], c=c)
    ax_jtot.plot(t[0] / 3600.0, jtot[0] / 1e16, c=c)
    # gray -= dgray
    ax_jpos.set_xlim([0,45])
    ax_jpos.set_ylim([0,150])
    ax_jvel.set_ylim([0,2.1])
    ax_jtot.set_ylim([0,50])
    return 1.0 * n_lost / m

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
