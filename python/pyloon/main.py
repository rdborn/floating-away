import numpy as np
import cProfile
import datetime
import os
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
fieldestimator = 'gpfe'

# Choose the way to set up the wind field
# options: noaa, sounding, sine, brownsine, brownsounding
environment = 'noaa'

# IF environment = 'sounding' or 'brownsounding' set the sounding file to read from
file = './weather-data/oak_2017_07_01_00z.txt'
# file = './weather-data/oak_2018_02_08_00z.txt'

# IF environment = 'sine' or 'brownsine' set the parameters of the sine wave
resolution =	100
frequency =		2.0*np.pi/8000.0
amplitude =		30.0
phase =			0.0
offset =		0.0

# IF environment = 'noaa' set the lat/lon center (origin) and span of the field
origin =	np.array([37.4268, -122.1733])
latspan =	60000.0
lonspan =	60000.0

# Upper and lower altitude bounds for the balloon
lower = 5000.0
upper = 28000.0

# Min and max altitude for which there is flow data
zmin = 0.0
zmax = 30000.0

# Choose parameters for identifying jetstreams in the wind field
streamres =		200
streammin =		0.0
streammax =		30000.0
streamsize =	10
threshold =		0.1

# Balloon initial position
xi = 0.0
yi = 0.0
zi = 10.0

# Simulation sampling frequency
Fs = 0.2

# Boolean indicating whether to enable calling of the LoonSim.plot() function
plot = True
plottofile = True
out_plot_folder = './fig/fig_' + \
				planner + '_' + \
				fieldestimator + '_' + \
				environment + '_' + \
				datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + '/'

# Set point
pstar = np.array([0.0, 0.0, 17500.0])

# Control effort
u = 5.0

# Duration of "zero control effort"
T = 180.0

# MPC depth
depth = 7

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
			plot=plot,
			plottofile=plottofile)

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

if plot:
	if plottofile:
		if not os.path.exists(out_plot_folder):
			os.mkdir(out_plot_folder)
		out_plot_folder = out_plot_folder + 'png/'
		if not os.path.exists(out_plot_folder):
			os.mkdir(out_plot_folder)
pos = LS.loon.get_pos()
buffer = 30.0
n_nodes_desired = 256
it = 0
n_frames = 100
while it < n_frames:
	out_file = out_plot_folder + str(it).zfill(len(str(n_frames))) + '.png'
	it += 1
	n_jets = 0
	for jet in LS.pathplanner.planner.jets.jetstreams.values():
		if jet.avg_alt > lower and jet.avg_alt < upper:
			n_jets += 1
	depth = np.int(np.ceil(np.log(n_nodes_desired) / np.log(n_jets)))
	print(str(it) + " Planning... depth: " + str(depth))
	pol = LS.plan(	u=u,
					T=T,
					pstar=pstar,
					depth=depth)
	print("Policy:")
	print("\t" + str(pol))
	# LS.pathplanner.planner.plot()
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
		if plottofile:
			LS.plot(outfile=out_file)
		else:
			LS.plot()
