import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

import numpy as np
import copy
from cvxopt import matrix, solvers

# minimize  || y - Ax ||_2
# st        Cx = b
#           x >= 0
#


threshold = 1e-3
alpha = 1e-3

consider_std = True

magnitude = np.array([11., 12., 26., 7., 25., 9.])
direction = np.array([37., 49., 80., 270., 94., 102.]) * np.pi / 180.0
vx = magnitude * np.cos(direction)
vy = magnitude * np.sin(direction)
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
y = np.matrix([0.,0.,0.,0.]).T
I = np.matrix(np.eye(A.shape[1]))
Z = np.matrix(np.zeros(A.shape[1])).T

_P = matrix(2 * A.T * A)
_q = matrix((-2 * y.T * A).T)
_G = matrix(-I)
_h = matrix(Z)
_A = matrix(C)
_b = matrix(b)

sol = solvers.qp(_P, _q, _G, _h, _A, _b)
x = np.squeeze(np.array(sol['x']))
# x[x < 0.05] = 0.0
# x = x / np.linalg.norm(x)
for i, xi in enumerate(x):
    print("\t" + str(np.int(x[i]*100)).zfill(2) + "% at\t" + str(np.int(vx[i])) + ", " + str(np.int(vy[i])))
print("\tOptimal vel:\t" + str(np.dot(x,vx)) + ", " + str(np.dot(x,vy)))
print("\tOptimal std:\t" + str(np.dot(x,stdx)) + ", " + str(np.dot(x,stdy)))



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
