import numpy as np
from scipy.stats import norm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def pfields(label, values):
    msg = "{:10}".format(label) + "|"
    divider = "\n-----------"
    for v in values:
        msg = msg + "{:>10}".format(v) + " |"
        divider = divider + "------------"
    msg = msg + divider
    print(msg)
    return msg

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

def grid_uneven_data(data, tidx, T, N):
    data_grid = []
    for dataset in data:
        t = dataset[tidx]
        t_sim_end = t[-1]
        t_grid = np.linspace(0, T, N)
        t_lost_idx = (t_grid > t_sim_end)
        dataset_grid = []
        for i, field in enumerate(dataset):
            if i != tidx:
                field_grid = griddata(np.array(t), np.array(field), t_grid, method='cubic')
                field_grid[t_lost_idx] = 1000000#field[-1]
                dataset_grid.append(field_grid)
            else:
                dataset_grid.append(t_grid)
        data_grid.append(dataset_grid)
    return data_grid

def breakpoint():
    raw_input("BREAKPOINT (press ENTER to continue)")

def combined_mean(arr1, arr2):
    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    n1 = len(arr1)
    n2 = len(arr2)
    muc = (mu1 * n1 + mu2 * n2) / (n1 + n2)
    return muc

def combined_var(arr1, arr2):
    var1 = np.var(arr1)
    var2 = np.var(arr2)
    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    n1 = len(arr1)
    n2 = len(arr2)
    if n1 == 0:
        return var2
    if n2 == 0:
        return var1
    muc = (mu1 * n1 + mu2 * n2) / (n1 + n2)
    varc = (n1 * (var1 + (mu1 - muc)**2) + n2 * (var2 + (mu2 - muc)**2)) / (n1 + n2)
    return varc

def bivar_normal_entropy(stdx, stdy):
    # This is for a bivariate normal distribution with no covariance
    return 0.5 * np.log((2 * np.pi * np.e)**2 * stdx * stdy)

def multivar_normal_entropy(Sigma):
    return 0.5 * np.log((2 * np.pi * np.e)**(Sigma.shape[0]) * np.linalg.det(Sigma))

def get_samesign_bounds(x, stdx):
    if len(np.squeeze(np.array(x)).shape) == 0:
        xmax = np.sign(x) * np.max([np.sign(x) * (x + stdx), 0.0])
        xmin = np.sign(x) * np.max([np.sign(x) * (x - stdx), 0.0])
    else:
        x = np.array(x)
        stdx = np.array(stdx)
        xmax = np.sign(x) * (x + stdx)
        xmin = np.sign(x) * (x - stdx)
        xmax[xmax < 0] = 0
        xmin[xmin < 0] = 0
        xmax = np.sign(x) * xmax
        xmin = np.sign(x) * xmin
    return xmin, xmax

def dist_weights(d):
    inv_d = 1.0 / d
    sum_inv_d = np.sum(inv_d)
    weights = inv_d / sum_inv_d
    return weights

def get_angle_range(x, y, stdx, stdy):
    theta_nom = rad2deg(np.arctan2(y, x))
    xmin, xmax = get_samesign_bounds(x, stdx)
    ymin, ymax = get_samesign_bounds(y, stdy)

    if stdx < abs(x):
        if stdy < abs(y):
            theta_upper = rad2deg(np.arctan2(y + np.sign(x) * stdy, x - np.sign(y) * stdx))
            theta_lower = rad2deg(np.arctan2(y - np.sign(x) * stdy, x + np.sign(y) * stdx))
        else:
            theta_upper = rad2deg(np.arctan2(y + np.sign(x) * stdy, x - np.sign(x) * stdx))
            theta_lower = rad2deg(np.arctan2(y - np.sign(x) * stdy, x - np.sign(x) * stdx))
    else:
        if stdy < abs(y):
            theta_upper = rad2deg(np.arctan2(y - np.sign(y) * stdy, x - np.sign(y) * stdx))
            theta_lower = rad2deg(np.arctan2(y - np.sign(y) * stdy, x + np.sign(y) * stdx))
        else:
            theta_upper = theta_nom + 180.0
            theta_lower = theta_nom - 180.0
    theta_lower = theta_lower if theta_lower < theta_nom else theta_lower - 360.0
    theta_upper = theta_upper if theta_upper > theta_nom else theta_upper + 360.0
    return theta_lower, theta_nom, theta_upper

def get_angle_dist(x, y, stdx, stdy, n):
    x = np.random.normal(x, stdx, n)
    y = np.random.normal(y, stdy, n)
    theta = np.arctan2(y, x)
    # theta = np.zeros(n)
    # for i in range(len(theta)):
    #     x_i = np.random.normal(x, stdx)
    #     y_i = np.random.normal(y, stdy)
    #     theta[i] = np.arctan2(y_i, x_i)
    # theta = theta % (2 * np.pi)
    # print(np.array(theta*180/np.pi,dtype=int))
    theta_mu_1 = np.mean(theta)
    theta_std_1 = np.std(theta)
    # print(np.array([theta_mu, theta_std])*180/np.pi)
    theta[theta>np.pi] -= 2*np.pi
    # print(np.array(theta*180/np.pi,dtype=int))
    theta_mu_2 = np.mean(theta)
    theta_std_2 = np.std(theta)
    # print(np.array([theta_mu, theta_std])*180/np.pi)
    # if theta_std_1 < theta_std_2:
        # return theta_mu_1, theta_std_1
    # return theta_mu_2, theta_std_2
    return theta_mu_1, theta_std_1

def get_mag_dist(x, y, stdx, stdy, n):
    mag = np.zeros(n)
    for i in range(len(mag)):
        x_i = np.random.normal(x, stdx)
        y_i = np.random.normal(y, stdy)
        mag[i] = np.sqrt(x_i**2 + y_i**2)
    mag_mu = np.mean(mag)
    mag_std = np.std(mag)
    return mag_mu, mag_std

def rad2deg(theta):
    return (theta * 180.0 / np.pi) % 360.0

def continuify_angles(theta):
    # return theta
    buf = 190
    for i in range(len(theta)):
        if i > 0:
            if abs(theta[i] - theta[i-1]) > buf:
                if theta[i] > theta[i-1]:
                    theta[i] = theta[i] - 360
                else:
                    theta[i] = theta[i] + 360
    return theta

def compare(x1, x2):
    compare_1 = np.array(x1)
    compare_2 = np.array(x2)
    comparison = np.atleast_1d(np.squeeze(np.array([compare_1 == compare_2])))
    return comparison

def parsekw(kwargs, kw, default):
    return (kwargs.get(kw) if not (compare(kwargs.get(kw), None)).any() else default)

def hash3d(p):
    P0 = 73856093
    P1 = 19349663
    P2 = 83492791
    return (int(p[0]*P0) ^ int(p[1]*P1) ^ int(p[2]*P2))

def hash4d(p):
    P0 = 73856093
    P1 = 19349663
    P2 = 83492791
    P3 = 32452843
    return (int(p[0]*P0) ^ int(p[1]*P1) ^ int(p[2]*P2) ^ int(p[3]*P3))

def rng(c):
    # return 2 * (0.5 - np.random.rand()) * c
    return norm.rvs(scale=c)

def vector_sum(mag1, dir1, mag2, dir2):
    xcomp = mag1 * np.cos(dir1) + mag2 * np.cos(dir2)
    ycomp = mag1 * np.sin(dir1) + mag2 * np.sin(dir2)
    mag = np.sqrt(xcomp**2 + ycomp**2)
    angle = np.arctan2(ycomp, xcomp)
    return mag, angle

def warning(warning):
    print("WARNING: " + warning)
    return False

def downsize(M):
    mask = np.zeros(len(M[0]), dtype=bool)
    for i in range(len(M)):
        if i > 0:
            mask |= (M[i-1] != M[i])
        if all(mask):
            return mask
    return mask

def normalize(x):
    mu = np.mean(x)
    sigma = np.sqrt(np.var(x))
    return (x - mu) / sigma

def saturate(x, sat):
    if np.isnan(x):
        warning("Cannot saturate NaN. Returning NaN.")
        return x
    sat = abs(sat)
    x = x if x < sat else sat
    x = x if x > -sat else -sat
    return x
