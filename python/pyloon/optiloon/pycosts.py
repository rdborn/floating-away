import numpy as np
import copy

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng, downsize
from pyutils import pyutils

def J_position(p, pstar):
    p = parsekw(kwargs, 'p', None)
    pstar = parsekw(kwargs, 'pstar', None)

    J_position = np.linalg.norm(p - pstar)
    return J_position

def J_velocity(p, pstar, pdot):
    p = parsekw(kwargs, 'p', None)
    pstar = parsekw(kwargs, 'pstar', None)
    pdot = parsekw(kwargs, 'pdot', None)

    pdothat = pdot / np.linalg.norm(pdot)
    phi = p - pstar
    norm_phi = np.linalg.norm(phi)
    phihat = phi / norm_phi if norm_phi > 0 else phi
    J_velocity = (np.dot(phihat, pdothat)+1)
    return J_velocity

def J_acceleration(p, pstar, pdot):
    p = parsekw(kwargs, 'p', None)
    pstar = parsekw(kwargs, 'pstar', None)
    pdot = parsekw(kwargs, 'pdot', None)
    
    norm_p = np.linalg.norm(p)
    phat = p / norm_p if norm_phi > 0 else phi
    phidot = np.dot(phat, pdot) * phat
    phiddot = (((np.linalg.norm(pdot)**2 - 2 * np.linalg.norm(phidot)**2)) * phat + np.linalg.norm(phidot) * pdot)
    phiddot = phiddot / norm_p if norm_p > 0 else phiddot
    J_acceleration = np.linalg.norm(phiddot)
    return J_acceleration
