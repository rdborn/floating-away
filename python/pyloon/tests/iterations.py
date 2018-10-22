import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

import matplotlib.pyplot as plt

import numpy as np
import copy
from cvxopt import matrix, solvers

# minimize  || y - Ax ||_2
# st        Cx = b
#           x >= 0
#


threshold = 1e-3
alpha = 1e-3

consider_std = False

magnitude = np.array([11., 12., 26., 7., 25., 9.])
direction = np.array([37., 49., 80., 270., 94., 102.]) * np.pi / 180.0
vx = np.cos(direction)
vy = np.sin(direction)
stdx = np.array([0.1, 0.3, 0.1, 0.5, 0.02, 0.3])
stdy = np.array([0.02, 0.3, 0.2, 0.1, 0.05, 0.2])
# stdx = 0.005**2 * np.ones(len(vx))
# stdy = 0.005**2 * np.ones(len(vy))
if consider_std:
    sx = stdx
    sy = stdy
else:
    sx = np.zeros(len(vx))
    sy = np.zeros(len(vy))
A = np.matrix([vx, vy, sx, sy])
b = np.matrix(1.)
C = np.matrix(np.ones(A.shape[1]))
p = np.matrix(np.eye(A.shape[1]))
y = np.matrix([4.,1.,0.,0.]).T
I = np.matrix(np.eye(A.shape[1]))
Z = np.matrix(np.zeros(A.shape[1])).T

_P = matrix(2 * A.T * A)
_q = matrix((-2 * y.T * A).T)
# _q = matrix((-2 * (C * A.T * A + y.T * A)).T)
_G = matrix(-I)
_h = matrix(Z)
_A = matrix(C)
_b = matrix(b)
_A = None
_b = None

plotting = False
if plotting:
    L = np.logspace(-5,5,1000)
    x_all = np.zeros([len(L), A.shape[1]])
    y_all = np.zeros([len(L), len(y)])
    y_all = np.zeros([len(L), 2])
    y_all_approx = np.zeros([len(L), 2])
    fig = plt.figure()
    ax = plt.gca()
    for i, l in enumerate(L):
        # _b = matrix(l)
        # _P = matrix(2 * A.T * A + 0. * I)
        _P = matrix(2 * (A - y * C).T * (A - y * C) + l * I)
        # _q = matrix((-2 * y.T * A + 0. * C).T)
        _q = matrix((0. * C).T)
        sol = solvers.qp(_P, _q, _G, _h, _A, _b)
        x = np.squeeze(np.array(sol['x']))
        x_all[i] = x / np.sum(x)
        x_hat = x / np.sum(x)
        y_all[i] = np.squeeze(y[0:2]) - np.array([np.dot(x_hat,vx), np.dot(x_hat,vy)])
        x_hat[x_all[i] < 1e-2] = 0.
        y_all_approx[i] = np.squeeze(y[0:2]) - np.array([np.dot(x_hat,vx), np.dot(x_hat,vy)])
        # y_all[i] = np.squeeze(y) - np.array([np.dot(x,vx), np.dot(x,vy), np.dot(x,stdx), np.dot(x,stdy)])
        # if np.sum(x > 1e-5) < 4:
        #     print(x / np.sum(x))
        #     print(y_all[i])
        #     idx_lim = i
        #     break
    for j in range(x_all.shape[1]):
        ax.semilogx(L, x_all[:,j])
    ax2 = ax.twinx()
    for j in range(y_all.shape[1]):
        ax2.semilogx(L, y_all[:,j], '--')
        ax2.semilogx(L, y_all_approx[:,j], ':')
    ax.set_ylim([0,1.1])
    ax2.set_ylim([-5,5])
    # ax.set_xlim([-1e10,L[idx_lim]])
    # ax.set_xlim([-1e10,L[y_all > 0.99*np.max(y_all)][0]])
    plt.show()
else:
    # _P = matrix(2 * (A - y * C).T * (A - y * C))
    # _P = matrix(2 * (A.T * A))
    # _q = matrix((0 * C).T)
    # _q = matrix((0 * C).T)
    sol = solvers.qp(_P, _q, _G, _h, _A, _b)
    x = np.squeeze(np.array(sol['x']))
    x /= np.sum(x)
    x_hat = x #/ np.sum(x)
    # x[x_hat < 1e-5] = 0.
    # x[x < 0.05] = 0.0
    # x = x / np.linalg.norm(x)
    for i, xi in enumerate(x):
        print("\t" + str(np.int(x_hat[i]*100)).zfill(2) + "% at\t" + str(np.int(vx[i]*100)) + ", " + str(np.int(vy[i]*100)))
    print("\tOptimal vel:\t" + str(np.dot(x,vx)) + ", " + str(np.dot(x,vy)))
    print("\tOptimal std:\t" + str(np.dot(x,stdx)) + ", " + str(np.dot(x,stdy)))
    print(np.sqrt(np.dot(x,vx)**2 + np.dot(x,vy)**2))
    print(np.arctan2(np.dot(x,vy), np.dot(x,vx))*180/np.pi)
    print(np.arctan2(y[1], y[0])*180/np.pi)




# def dfdx(x):
#     x = np.matrix(x)
#     return ((y - A * x) / np.linalg.norm(y - A * x)).T * (-A)
#
# def f(x):
#     return np.linalg.norm(y - A * x)
#
# x = np.matrix(np.ones(A.shape[1])).T
# x = x / np.linalg.norm(x)
# fx_curr = f(x)
# fx_prev = np.inf
#
# while abs(fx_curr - fx_prev) > threshold:
#     fdot = dfdx(x).T
#     q = fx_curr / fdot
#     r = C
#     for row in p:
#         if row * x <= 0:
#             print(r)
#             print(row)
#             r = np.cross(r, row)
#             # print("woah")
#             # print(q.T)
#             # q -= np.squeeze(np.array(row * q)) * row.T / np.linalg.norm(row)**2
#             # print(q.T)
#     q -= np.squeeze(np.array(r * q)) * r.T / np.linalg.norm(r)**2
#     x -= alpha * q
#     fx_prev = copy.deepcopy(fx_curr)
#     fx_curr = f(x)
#     print(np.array(100*x.T,dtype=int))
#     print(C*x)
# print(C*x)
