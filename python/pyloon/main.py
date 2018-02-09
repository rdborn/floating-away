from loonsim import LoonSim
from optiloon.optiloon import LoonPathPlanner
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
from skewt import SkewT

# Set up simulation parameters
xdim = 100
ydim = 100
zdim = 100
hz = 0.2
duration = 6

# Set up flow field
file = "./weather-data/oak_2017_07_01_00z.txt"
LS = LoonSim(file=file, Fs=hz)

# Set point
pstar = [0.0, 0.0, 15000.0]

# Set up the plot
myplot = plt.figure().gca(projection='3d')
plt.ion()

last_pos = LS.loon.get_pos()
pos = last_pos
lo = LS.loon.z - 5
hi = LS.loon.z + 5
LPP = LoonPathPlanner(field=LS.field, res=3, lo=10000, hi=30000, sounding=True)
thresh = 3

# Simulation
i = 0
while(True):
	i += 1
	print("Monte carlo numba: " + str(i))

	# Set up and train path planner
	# TODO: reuse the same path planner for as long as possible
	# if np.linalg.norm(np.subtract(pos, last_pos)) > thresh:
	# 	lo = LS.loon.z - 10
	# 	hi = LS.loon.z + 10
	# 	LPP = LoonPathPlanner(field=LS.field, res=3, lo=lo, hi=hi, sounding=True)

	# Monte Carlo through the wind field using our path planner
	depth = 4
	LPP.montecarlo(LS.loon, pstar, depth)

	# Extract the optimal policy found by our path planner
	pol = LPP.policy()

	# Reset the path planner for reuse
	LPP.reset()

	# Implement the first steps of the optimal policy
	N = 1; # number of steps to take
	for j in range(N):
		print("Control effort: " + str(pol[-j-1]))
		for t in range(np.int(np.ceil(LPP.branch_length*LS.Fs))):
			LS.propogate(pol[-j-1]) # move the balloon along
			pos = LS.loon.get_pos() # get balloon's position
			myplot.scatter(pos[0],pos[1],pos[2]) # plot the position
	plt.pause(0.0001) # redraw the plot
	print(pos)

while(True):
	plt.pause(0.05)


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
