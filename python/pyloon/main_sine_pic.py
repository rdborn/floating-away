############################################
# DEPRECATED
############################################

from loonsim import LoonSim
from optiloon.loonpathplanner import MonteCarloPlanner as MCP
from optiloon.loonpathplanner import PlantInvertingController as PIC
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
from skewt import SkewT

# Set up simulation parameters
hz = 0.2
duration = 6

# Set up flow field
LS = LoonSim(	zmin=0.0,
			 	zmax=30000.0,
				resolution=100,
				frequency=2.0*np.pi/8000.0,
				amplitude=30.0,
				phase=0.0,
				offset=0.0,
				Fs=hz, xi=10000, yi=10000, zi=20500)

# Set point
pstar = [0.0, 0.0, 13000.0]

last_pos = LS.loon.get_pos()
pos = last_pos
lo = LS.loon.z - 5
hi = LS.loon.z + 5
LPP = PIC(field=LS.field, res=3, lower=10000, upper=30000)
thresh = 3

# Simulation
i = 0
while(True):
	i += 1

	pol = LPP.plan(loon=LS.loon)

	# Implement the first steps of the optimal policy
	N = 1; # number of steps to take
	print("Control effort: " + str(pol))
	for i in range(10):
		LS.propogate(pol) # move the balloon along
	pos = LS.loon.get_pos() # get balloon's position
	LS.plot()
	print(pos)

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
# LPP = LoonPathPlanner(field=LS.field, res=3, lo=lo, hi=hi, sounding=True)
#
# #z = np.linspace(0,zdim-1,1000)
# sorted_keys = np.sort(LS.field.field.keys())
# z = np.linspace(sorted_keys[0],sorted_keys[-1],2000)
# fx = np.zeros(len(z))
# fx_actual = np.zeros(len(z))
# fx_std = np.zeros(len(z))
# fy = np.zeros(len(z))
# fy_actual = np.zeros(len(z))
# fy_std = np.zeros(len(z))
# for i in range(len(z)):
# 	result = LPP.predict(z[i])
# 	mag, angle = LS.field.get_flow([(lo+hi)/2.0,(lo+hi)/2.0,z[i]])
# 	fx[i] = result[0]
# 	fx_actual[i] = mag * np.cos(angle) if abs(mag) < 50 else 0
# 	fx_std[i] = result[3]
# 	fy[i] = result[1]
# 	fy_actual[i] = mag * np.sin(angle) if abs(mag) < 50 else 0
# 	fy_std[i] = result[4]
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
