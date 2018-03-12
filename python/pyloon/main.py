import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt

from pysim.loonsim import LoonSim

#############################################
# SETUP PARAMETERS
#############################################

# Choose the planner algorithm and estimator scheme
# options: montecarlo, mpc, mpcfast, ldp, wap, pic
planner = 'mpcfast'
# options: gpfe, knn1dgp, multi1dgp
fieldestimator = 'multi1dgp'

# Choose the way to set up the wind field
# options: noaa, sounding, sine, brownsine, brownsounding
environment = 'noaa'

# IF environment = 'sounding' or 'brownsounding' set the sounding file to read from
file = './weather-data/oak_2017_07_01_00z.txt'

# IF environment = 'sine' or 'brownsine' set the parameters of the sine wave
resolution =	100
frequency =		2.0*np.pi/8000.0
amplitude =		30.0
phase =			0.0
offset =		0.0

# IF environment = 'noaa' set the lat/lon center (origin) and span of the field
origin =	np.array([37, -121])
latspan =	60000.0
lonspan =	60000.0

# Upper and lower altitude bounds for the balloon
lower = 5000.0
upper = 28000.0

# Min and max altitude for which there is flow data
zmin = 0.0
zmax = 30000.0

# Choose parameters for identifying jetstreams in the wind field
streamres =		500
streammin =		0.0
streammax =		30000.0
streamsize =	10
threshold =		0.001

# Balloon initial position
xi = 0.0
yi = 0.0
zi = 15000.0

# Simulation sampling frequency
Fs = 0.2

# Boolean indicating whether to enable calling of the LoonSim.plot() function
plot = True

# Set point
pstar = np.array([0.0, 0.0, 17500.0])

# Control effort
u = 5.0

# Duration of "zero control effort"
T = 180.0

# MPC depth
depth = 3

# Number of MPC steps to execute after planning
N = 1

#############################################
# SETUP SIMULATION
#############################################

# Set up simulation
LS = LoonSim(planner=planner,
			environment=environment,
			fieldestimator=fieldestimator,
			lower=lower,
			upper=upper,
			streamres=streamres,
			streammax=streammax,
			streammin=streammin,
			streamsize=streamsize,
			threshold=threshold,
			file=file,
			origin=origin,
			latspan=latspan,
			lonspan=lonspan,
			zmin=zmin,
			zmax=zmax,
			resolution=resolution,
			frequency=frequency,
			amplitude=amplitude,
			phase=phase,
			offset=offset,
			xi=xi,
			yi=yi,
			zi=zi,
			Fs=Fs,
			plot=plot)

#############################################
# LOCAL FUNCTIONS
#############################################

def choose_ctrl(x, xstar, buffer):
	if xstar - x > buffer:
		return 1
	elif xstar - x < -buffer:
		return -1
	else:
		return 0

#############################################
# SIMULATION
#############################################

pos = LS.loon.get_pos()
buffer = 30.0
while(True):
	print("Planning...")
	pol = LS.plan(	u=u,
					T=T,
					pstar=pstar,
					depth=depth)
	print("Policy:")
	print("\t" + str(pol))
	for i in range(N):
		print("Moving to altitude:")
		print("\t" + str(np.int(pol[i])))
		c = choose_ctrl(pos[2], pol[i], buffer)
		if c == 0:
			dt = 0.0
			while (T - dt) > 0.0:
				LS.propogate(c*u)
				dt += 1.0 / LS.loon.Fs
		else:
			while (pol[i] - pos[2]) * c * u > 0:
				LS.propogate(c*u)
				pos = LS.loon.get_pos()
	pos = LS.loon.get_pos()
	print("New position:")
	print("\t(" + str(np.int(pos[0])) + ", " + str(np.int(pos[1])) + ", " + str(np.int(pos[2])) + ")")
	if plot:
		LS.plot()
