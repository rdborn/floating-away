from loonsim import LoonSim
from optiloon.optiloon import LoonPathPlanner
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt

# Set up simulation parameters
xdim = 100
ydim = 100
zdim = 100
hz = 10
duration = 6

# Set up flow field
LS = LoonSim(xdim=xdim, ydim=ydim, zdim=zdim, Fs=hz)
for i in range(zdim):
	LS.field.set_planar_flow(int(i),1*np.sin(0.25*i),0.0)

# Set point
pstar = [50,30,60]

# Set up the plot
myplot = plt.figure().gca(projection='3d')
plt.ion()

# Simulation
for i in range(15):
	print("Monte carlo numba: " + str(i))

	# Set up and train path planner
	# TODO: reuse the same path planner for as long as possible
	lo = LS.loon.z - 15
	hi = LS.loon.z + 15
	LPP = LoonPathPlanner(field=LS.field, res=4, lo=lo, hi=hi)

	# Monte Carlo through the wind field using our path planner
	LPP.montecarlo(LS.loon, pstar, 3)

	# Extract the optimal policy found by our path planner
	pol = LPP.policy()

	# Reset the path planner for reuse
	LPP.reset()

	# Implement the first steps of the optimal policy
	N = 1; # number of steps to take
	for j in range(N):
		print("Control effort: " + str(pol[-j-1]))
		for t in range(LPP.branch_length*LS.Fs):
			LS.propogate(pol[-j-1]) # move the balloon along
			pos = LS.loon.get_pos() # get balloon's position
			myplot.scatter(pos[0],pos[1],pos[2]) # plot the position
	plt.pause(0.0001) # redraw the plot

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


z = np.linspace(0,zdim-1,1000)
fx = np.zeros(len(z))
fx_actual = np.zeros(len(z))
fx_std = np.zeros(len(z))
for i in range(len(z)):
	result = LPP.predict([(lo+hi)/2.0,(lo+hi)/2.0,z[i]])
	mag, angle = LS.field.get_flow((lo+hi)/2.0,(lo+hi)/2.0,z[i])
	fx[i] = result[0]
	fx_actual[i] = mag * np.cos(angle)
	fx_std[i] = result[3]

fig = plt.figure()
plt.plot(z, fx_actual, 'r:')
plt.plot(z, fx, 'b-')
plt.fill(np.concatenate([z,z[::-1]]), np.concatenate([fx - 3*fx_std, (fx + 3*fx_std)[::-1]]), alpha=0.5, fc='b', ec='None')
plt.show()
