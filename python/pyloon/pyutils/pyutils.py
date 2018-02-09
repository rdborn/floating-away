import numpy as np

def parsekw(kwargs, kw, default):
    return (kwarg.get(kw) if all(np.array(kwargs,get(kw)) != np.array(None)) else default)

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
    return 2 * (0.5 - np.random.rand()) * c

def vector_sum(mag1, dir1, mag2, dir2):
    xcomp = mag1 * np.cos(dir1) + mag2 * np.cos(dir2)
    ycomp = mag1 * np.sin(dir1) + mag2 * np.sin(dir2)
    mag = np.sqrt(xcomp**2 + ycomp**2)
    angle = np.arctan2(ycomp, xcomp)
    return mag, angle
