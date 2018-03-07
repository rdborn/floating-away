from loonsim import LoonSim
from optiloon.loonpathplanner import MonteCarloPlanner as MCP
from optiloon.loonpathplanner import PlantInvertingController as PIC
from optiloon.loonpathplanner import MPCWAPFast as WAP
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
from matplotlib.mlab import griddata
from skewt import SkewT

# Set up simulation parameters
hz = 0.2
duration = 6

# Set up flow field
LS = LoonSim(	origin=np.array([37.5, -120.5]),
				latspan=600.0,
				lonspan=600.0,
				Fs=hz,
				xi=0.0,
				yi=0.0,
				zi=15000.0,
				plot=True,
				noaa=True)

# Set point
pstar = np.array([0.0, 0.0, 17500.0])

last_pos = LS.loon.get_pos()
pos = last_pos
LPP = WAP(	field=LS.field,
			lower=12000,
			upper=20000,
			streamsize=10,
			threshold=0.001)
LPP.__delta_p_between_jetstreams__(5.0)

# altitude = 2000.0
# while True:
# 	N = 100
# 	M = 1500000.0
# 	     550000
# 	p = np.linspace(-M, M, N)
# 	_x = np.zeros(N**2)
# 	# _x = np.zeros(N)
# 	_y = np.zeros(N**2)
# 	# _y = np.zeros(N)
# 	_c = np.zeros(N**2)
# 	# _c = np.zeros(N)
# 	wind_plot = plt.figure().gca()
# 	for i, x in enumerate(p):
# 		# pos = np.array([0, 0, x])
# 		# vx = LPP.GPx.predict(np.atleast_2d(pos), return_std=False)
# 		# vy = LPP.GPy.predict(np.atleast_2d(pos), return_std=False)
# 		# magnitude = np.sqrt(vx[0][0]**2 + vy[0][0]**2)
# 		# _x[i] = x
# 		# _y[i] = magnitude
# 		for j, y in enumerate(p):
# 			pos = np.array([x, y, altitude])
# 			vx = LPP.GPx.predict(np.atleast_2d(pos), return_std=False)
# 			vy = LPP.GPy.predict(np.atleast_2d(pos), return_std=False)
# 			magnitude = np.sqrt(vx[0][0]**2 + vy[0][0]**2)
# 			_x[i*N + j] = x
# 			_y[i*N + j] = y
# 			_c[i*N + j] = magnitude
# 	# plt.plot(_x, _y)
# 	# plt.show()
# 	xplot = np.linspace(np.min(_x), np.max(_x), 100)
# 	yplot = np.linspace(np.min(_y), np.max(_y), 100)
# 	X, Y = np.meshgrid(xplot, yplot)
# 	Z = griddata(_x, _y, _c, xplot, yplot, interp='linear')
# 	plt.contourf(X, Y, Z)
# 	plt.colorbar()
# 	print(altitude)
# 	plt.show()
# 	altitude += 2000.0

# Simulation
while(True):
	u = 5.0
	T = 180.0
	depth = 3
	N = 1
	pol = LPP.plan(LS.loon, u, T, pstar, depth)
	pos = LS.loon.get_pos() # get balloon's position
	buffer = 30.0
	print(pos)
	print(pol)
	for i in range(N):
		u = 5.0
		if pol[i] - pos[2] > buffer:
			u = u
		elif pol[i] - pos[2] < -buffer:
			u = -u
		else:
			u = 0.0
		if u == 0:
			dt = 0.0
			while (T - dt) > 0.0:
				LS.propogate(u)
				dt += 1.0 / LS.loon.Fs
		else:
			while (pol[i] - pos[2]) * u > 0:
				LS.propogate(u)
				pos = LS.loon.get_pos()
	print(pos)
	LS.plot()

########################################
# PLOTTING
########################################

# leaves = DataFrame([row[0:4] for row in LPP.leaves], columns=['x','y','z','val'])
# leaves = leaves[1:]
# leaves_plot = plt.figure().gca(projection='3d')
# leaves_plot.scatter(leaves['x'],leaves['val'],leaves['z'])
#plt.show()

# lo = 3.0
# hi = 31000.0
# # LPP = LoonPathPlanner(field=LS.field, res=3, lo=lo, hi=hi, sounding=True)
# #
# z = np.linspace(lo,hi,1000)
# # sorted_keys = np.sort(LS.field.field.keys())
# # z = np.linspace(sorted_keys[0],sorted_keys[-1],1000)
# fx = np.zeros(len(z))
# fx_actual = np.zeros(len(z))
# fx_std = np.zeros(len(z))
# fy = np.zeros(len(z))
# fy_actual = np.zeros(len(z))
# fy_std = np.zeros(len(z))
# for i in range(len(z)):
# 	result = LPP.predict(np.array([0,0,z[i]]))
# 	mag, angle = LS.field.get_flow(p=np.array([0,0,z[i]]))
# 	fx[i] = result[0]
# 	fx_actual[i] = mag * np.cos(angle) if abs(mag) < 50 else 0
# 	fx_std[i] = result[2]
# 	fy[i] = result[1]
# 	fy_actual[i] = mag * np.sin(angle) if abs(mag) < 50 else 0
# 	fy_std[i] = result[3]
#
# fig = plt.figure()
# plt.plot(z, fx_actual, 'r:')
# plt.plot(z, fx, 'b-')
# plt.fill(np.concatenate([z,z[::-1]]), np.concatenate([fx - 3*fx_std, (fx + 3*fx_std)[::-1]]), alpha=0.5, fc='b', ec='None')
# fig = plt.figure()
# plt.plot(z, fy_actual, 'r:')
# plt.plot(z, fy, 'b-')
# plt.fill(np.concatenate([z,z[::-1]]), np.concatenate([fy - 3*fy_std, (fy + 3*fy_std)[::-1]]), alpha=0.5, fc='b', ec='None')
# plt.show()
#
# while(True):
# 	plt.pause(0.05)
