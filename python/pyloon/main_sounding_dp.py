from loonsim import LoonSim
from optiloon.loonpathplanner import MonteCarloPlanner as MCP
from optiloon.loonpathplanner import PlantInvertingController as PIC
from optiloon.loonpathplanner import MPCWAPFast as WAP
from optiloon.loonpathplanner import LocalDynamicProgrammingPlanner as LDPP
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
from skewt import SkewT

# Set up simulation parameters
hz = 0.2
duration = 6

# Set up flow field
file = "./weather-data/oak_2017_07_01_00z.txt"
# file = "./weather-data/oak_2018_02_08_00z.txt"
LS = LoonSim(file=file, Fs=hz, xi=0.0, yi=0.0, zi=10000.0, plot=True)

# Set point
pstar = np.array([0.0, 0.0, 17500.0])

last_pos = LS.loon.get_pos()
pos = last_pos

LPP = LDPP(	field=LS.field,
			lower=5000,
			upper=30000,
			streamsize=5,
			pstar=pstar,
			dp=np.array([100000, 100000, 50000]),
			dt=180.0,
			dims=np.array([10, 10, 10]),
			gamma=0.9)

# print("plotting...")
# LPP.plot(5.0, 180.0)
# while(True):
	# pass

# Simulation
while(True):
	u = 5.0
	T = 180.0
	pol = LPP.plan(LS.loon)
	pos = LS.loon.get_pos() # get balloon's position
	buffer = 100.0
	print(pos)
	print(pol)
	u = 5.0
	if pol - pos[2] > buffer:
		u = u
	elif pol - pos[2] < -buffer:
		u = -u
	else:
		u = 0.0
	if u == 0:
		dt = 0.0
		while (T - dt) > 0.0:
			LS.propogate(u)
			dt += 1.0 / LS.loon.Fs
	else:
		while (pol - pos[2]) * u > 0:
			LS.propogate(u)
			pos = LS.loon.get_pos()
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