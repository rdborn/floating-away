import numpy as np
import copy

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng, downsize
from pyutils import pyutils

def J_position(*args, **kwargs):
    p = np.array(parsekw(kwargs, 'p', None))
    pstar = np.array(parsekw(kwargs, 'pstar', None))

    J_pos = np.linalg.norm(p - pstar)
    return J_pos

def J_velocity(*args, **kwargs):
    p = np.array(parsekw(kwargs, 'p', None))
    pstar = np.array(parsekw(kwargs, 'pstar', None))
    pdot = np.array(parsekw(kwargs, 'pdot', None))

    pdothat = pdot / np.linalg.norm(pdot)
    phi = p - pstar
    norm_phi = np.linalg.norm(phi)
    phihat = phi / norm_phi if norm_phi > 0 else phi
    J_vel = (np.dot(phihat, pdothat)+1)
    # print J_vel
    return J_vel

def J_acceleration(*args, **kwargs):
    p = np.array(parsekw(kwargs, 'p', None))
    pstar = np.array(parsekw(kwargs, 'pstar', None))
    pdot = np.array(parsekw(kwargs, 'pdot', None))

    norm_p = np.linalg.norm(p)
    phat = p / norm_p if norm_p > 0 else p
    phidot = np.dot(phat, pdot) * phat
    phiddot = (((np.linalg.norm(pdot)**2 - 2 * np.linalg.norm(phidot)**2)) * phat + np.linalg.norm(phidot) * pdot)
    phiddot = phiddot / norm_p if norm_p > 0 else phiddot
    J_accel = np.linalg.norm(phiddot)
    return J_accel

def J_reachable_set(*args, **kwargs):
    p = np.array(parsekw(kwargs, 'p', None))
    pstar = np.array(parsekw(kwargs, 'pstar', None))
    jsi = parsekw(kwargs, 'jsi', None)
    p = p[0:2]
    pstar = pstar[0:2]

    phi = p - pstar
    norm_phi = np.linalg.norm(phi)
    phihat = phi / norm_phi if norm_phi > 0 else phi
    v_best = jsi.best_we_can_do([-phi[0], -phi[1]])
    norm_v = np.linalg.norm(v_best)
    vhat = v_best / norm_v if norm_v != 0 else v_best
    J_reachable = (np.dot(phihat, vhat)+1)
    print("Jr")
    print J_reachable
    return J_reachable

def J_drift(*args, **kwargs):
    p = np.array(parsekw(kwargs, 'p', None))
    pstar = np.array(parsekw(kwargs, 'pstar', None))
    dp = np.array(parsekw(kwargs, 'dp', None))
    p = p[0:2]
    pstar = pstar[0:2]
    dp = dp[0:2]

    phi = p - pstar
    norm_phi = np.linalg.norm(phi)
    phihat = phi / norm_phi if norm_phi > 0 else phi
    norm_dp = np.linalg.norm(dp)
    dphat = dp / norm_dp if norm_dp != 0 else dp
    J_d = (np.dot(phihat, dphat)+1)
    return J_d

def range_J(cost_function, n, *args, **kwargs):
    p_mu = np.array(parsekw(kwargs, 'p', None))
    pdot_mu = np.array(parsekw(kwargs, 'pdot', None))
    p_std = np.array(parsekw(kwargs, 'pstd', 1e-6))
    pdot_std = np.array(parsekw(kwargs, 'pdotstd', 1e-6))
    J = np.zeros(n)
    for i in range(n):
        kwargs['p'] = np.random.normal(p_mu, p_std)
        kwargs['pdot'] = np.random.normal(pdot_mu, pdot_std)
        J[i] = cost_function(**kwargs)
    J_mu = np.mean(J)
    J_std = np.std(J)
    return J_mu, J_std
