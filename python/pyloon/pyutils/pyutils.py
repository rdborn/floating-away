import numpy as np

def compare(x1, x2):
    compare_1 = np.array(x1)
    compare_2 = np.array(x2)
    comparison = np.atleast_1d(np.squeeze(np.array([compare_1 == compare_2])))
    return comparison

def parsekw(kwargs, kw, default):
    return (kwargs.get(kw) if not any(compare(kwargs.get(kw), None)) else default)

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
