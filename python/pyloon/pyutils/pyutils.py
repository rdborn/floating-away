import numpy as np
from scipy.stats import norm

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
