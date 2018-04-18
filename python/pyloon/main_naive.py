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
# options: montecarlo, mpc, mpcfast, ldp, wap, pic, naive
planner = 'naive'
gamma = 1e7 # tuning parameter for cost function (lower = care more about pos, less about vel)
# options: gpfe, knn1dgp, multi1dgp
fieldestimator = 'multi1dgp'

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
latspan =	140000.0
lonspan =	140000.0

# Upper and lower altitude bounds for the balloon
lower = 5000.0
upper = 22000.0

# IF planner = 'naive'
resamplethreshold = 500000
trusttime = 6

# Min and max altitude for which there is flow data
zmin = 0.0
zmax = 30000.0

# Choose parameters for identifying jetstreams in the wind field
streamres =		1000
streammin =		lower
streammax =		upper
M_PER_SAMPLE =	((streammax - streammin) / streamres)
streamsize = 	np.int(np.ceil(300 / M_PER_SAMPLE))
threshold =		0.005

# Balloon initial position
xi = 0.0
yi = 0.0
zi = 10.0

# Simulation sampling frequency
Fs = 0.2

# Boolean indicating whether to enable calling of the LoonSim.plot() function
plot = True
plottofile = True

# Set point
pstar = np.array([0.0, 0.0, 17500.0])

# Control effort
u = 5.0

# Duration of "zero control effort"
T = 300.0

# MPC depth
depth = 7

# Number of MPC steps to execute after planning
N = 1

#############################################
# SETUP SIMULATION
#############################################

# Set up simulation


#############################################
# LOCAL FUNCTIONS
#############################################

def choose_ctrl(x, xstar, buf):
	if xstar - x > buf:
		return 1
	elif xstar - x < -buf:
		return -1
	else:
		return 0

def get_there(LS, target, buf, u, T):
	pos = LS.loon.get_pos()
	p = pos[2]
	c = choose_ctrl(p, target, buf)
	if c == 0:
		dt = 0.0
		while (T - dt) > 0.0:
			LS.propogate(c*u)
			dt += 1.0 / LS.loon.Fs
	else:
		while (target - p) * c * u > 0:
			LS.propogate(c*u)
			p = LS.loon.get_pos()[2]
	return LS

def vel_cost(pos, pstar, v):
    p = pos[0:2]
    pstar = pstar[0:2]
    pdot = v
    pdothat = pdot / np.linalg.norm(pdot)
    norm_p = np.linalg.norm(p)
    pstar = pstar[0:2]
    phi = p - pstar
    norm_phi = np.linalg.norm(phi)
    phihat = phi / norm_phi if norm_phi > 0 else phi
    J_velocity = (np.dot(phihat, pdothat)+1)
    J = J_velocity
    return J

#############################################
# SIMULATION
#############################################

while True:
	for n_samples in range(2,11):
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
					plottofile=plottofile,
					resamplethreshold=resamplethreshold,
					trusttime=trusttime)

		points_to_sample = np.sort(np.random.uniform(lower, upper, n_samples))
		movie_name = planner + '_' + \
					str(n_samples) + '-samples_' + \
					environment + '_' + \
					datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
		record_name = movie_name
		out_plot_folder = './naives/fig_' + movie_name + '/'
		movie_folder = out_plot_folder
		record_folder = out_plot_folder


		if plot:
			if plottofile:
				if not os.path.exists(out_plot_folder):
					os.mkdir(out_plot_folder)
				with open((out_plot_folder + "README.txt"), "w+") as f:
					f.write("SIMULATION PARAMETERS\n" + \
						"Planner:\t" + planner + "\n" + \
						"Field Estimator:\t" + fieldestimator + "\n" + \
						"Environment Type:\t" + environment + "\n" + \
						"Minimum Safe Altitude:\t" + str(lower) + "\tm\n" + \
						"Maximum Safe Altitude:\t" + str(upper) + "\tm\n" + \
						"Initial X Position:\t" + str(xi) + "\tm from origin\n" + \
						"Initial Y Position:\t" + str(yi) + "\tm from origin\n" + \
						"Initial Z Position:\t" + str(zi) + "\tm from origin\n" + \
						"Gamma (cost tuning):\t" + str(gamma) + "\n" + \
						"\n" + \
						"NAIVE METHOD PARAMETERS (if applicable)\n" + \
						"Resample Threshold:\t" + str(resamplethreshold) + "\tm\n" + \
						"Trust Time:\t" + str(trusttime) + "\thr\n" + \
						"Number of Samples:\t" + str(n_samples) + "\n" +  \
						"\n" + \
						"JETSTREAM IDENTIFIER PARAMETERS (if applicable)\n" + \
						"Sampling Resolution:\t" + str(streamres) + "\tsamples\n" + \
						"Lower Altitude Bound:\t" + str(streammin) + "\tm\n" + \
						"Upper Altitude Bound:\t" + str(streammax) + "\tm\n" + \
						"Variance Threshold:\t" + str(threshold) + "\n" + \
						"Minimum Jetstream Size:\t" + str(streamsize) + "\tsamples\n" + \
						"\t(Meters per sample:\t" + str(M_PER_SAMPLE) + "\tm)\n" + \
						"\n" + \
						"NOAA FIELD PARAMETERS (if applicable)\n" + \
						"Origin (Latitude):\t" + str(origin[0]) + "\tdeg\n" + \
						"Origin (Longitude):\t" + str(origin[1]) + "\tdeg\n" + \
						"Span (Latitude):\t" + str(latspan) + "\tm\n" + \
						"Span (Longitude):\t" + str(lonspan) + "\tm\n" + \
						"\n" + \
						"SOUNDING FIELD PARAMETERS (if applicable)\n" + \
						"Sounding File:\t" + file + "\n" + \
						"\n" + \
						"SINE FIELD PARAMETERS (if applicable)\n" + \
						"Resolution:\t" + str(resolution) + \
						"Frequency:\t" + str(frequency) + \
						"Amplitude:\t" + str(amplitude) + \
						"Phase:\t" + str(phase) + \
						"Offset:\t" + str(offset) + \
						"\n" + \
						"TIME SIMULATION BEGAN\n" + \
						"Year:\t" + datetime.datetime.now().strftime('%Y') + "\n" + \
						"Month:\t" + datetime.datetime.now().strftime('%m') + "\n" + \
						"Day:\t" + datetime.datetime.now().strftime('%d') + "\n" + \
						"Hour:\t" + datetime.datetime.now().strftime('%H') + "\n" + \
						"Minute:\t" + datetime.datetime.now().strftime('%M') + "\n" + \
						"Second:\t" + datetime.datetime.now().strftime('%S') + "\n" + \
						"\n" + \
						"SIMULATION START TIME (in simulation)\n" + \
						"Year:\t" + "2017" + "\n" + \
						"Month:\t" + "07" + "\n" + \
						"Day:\t" + "01" + "\n" + \
						"Hour:\t" + "00" + "\n" + \
						"Minute:\t" + "00" + "\n" + \
						"Second:\t" + "00" + "\n" )
					f.write("\nNAIVELY SAMPLED ALTITUDES (if applicable):\n")
					for i, sample_alt in enumerate(np.sort(points_to_sample)):
						f.write(str(i) + "\t" + str(sample_alt) + "\n")
				out_plot_folder = out_plot_folder + 'png/'
				if not os.path.exists(out_plot_folder):
					os.mkdir(out_plot_folder)
		pos = LS.loon.get_pos()
		buf = 30.0
		n_nodes_desired = 150
		it = 0
		n_frames = 200
		while LS.tcurr < 46 * 3600:
			# Set output file for plots
			out_file = out_plot_folder + str(it).zfill(len(str(n_frames))) + '.png'
			# Increment iterator
			it += 1
			# Determine a tractable depth if we're using a tree search method
			if planner == 'mpcfast':
				n_jets = 0
				for jet in LS.pathplanner.planner.jets.jetstreams.values():
					if jet.avg_alt > lower and jet.avg_alt < upper:
						n_jets += 1
				depth = np.int(np.ceil(np.log(n_nodes_desired) / np.log(n_jets)))
			else:
				depth = 1
			print(str(it) + " Planning... depth: " + str(depth))
			# Figure out our next move
			pol = LS.plan(	u=u,
							T=T,
							pstar=pstar,
							depth=depth,
							gamma=gamma)
			pol = pol[1:] if len(pol) > 1 else pol
			print("Policy:")
			print("\t" + str(pol))
			# If the policy was negative, that was a sign from the naive path planner
			# that it needs to retrain, so we need to sample the air column
			if pol[0] < 0:
				# Figure out if we should go up or down first
				pos = LS.loon.get_pos()
				if abs(pos[2] - np.min(points_to_sample)) < abs(pos[2] - np.max(points_to_sample)):
					points_to_sample = np.sort(points_to_sample)
				else:
					points_to_sample = np.sort(points_to_sample)[::-1]
				# Go to the edge of the air column
				# Go to the other edge of the air column, sampling the field along the way
				for sample_alt in points_to_sample:
					LS = get_there(LS, sample_alt, buf, u, T)
					LS.sample()
				LS.pathplanner.planner.retrain()
			else:
				for i in range(N):
					if i >= len(pol):
						break
					print("Moving to altitude:")
					print("\t" + str(np.int(pol[i])))
					LS = get_there(LS, pol[i], buf, u, T)
			pos = LS.loon.get_pos()
			print("New position:")
			print("\t(" + str(np.int(pos[0])) + ", " + str(np.int(pos[1])) + ", " + str(np.int(pos[2])) + ")")
			# if plot:
			# 	if plottofile:
			# 		LS.plot(outfile=out_file, naive=(planner=='naive'))
			# 	else:
			# 		LS.plot(naive=(planner=='naive'))
			all_lats = np.array(LS.loon_history['x'][:])
			all_lons = np.array(LS.loon_history['y'][:])
			all_dists = np.sqrt(all_lats**2 + all_lons**2)
			all_t = LS.dt * np.array(range(len(all_dists)))
			avg_dists = np.cumsum(all_dists) / all_t
			if avg_dists[-1] > 20000:
				break

		# Write history to csv file
		t = LS.loon_history['t'][:]
		x = LS.loon_history['x'][:]
		y = LS.loon_history['y'][:]
		z = LS.loon_history['z'][:]
		vx = LS.loon_history['vx'][:]
		vy = LS.loon_history['vy'][:]
		vz = LS.loon_history['vz'][:]
		with open((record_folder + record_name + ".csv"), "w+") as f:
			f.write("jpos calculated as ||(x,y) - (xstar,ystar)||_2\n")
			f.write("jvel calculated as ((((x,y) - (xstar,ystar))_hat . (vx,vy)_hat) + 1), where ()_hat is a normalizing operator\n")
			f.write("jtot calculated as ln(jpos((jvel)(gamma)+1))\n")
			f.write("t,xstar,ystar,zstar,x,y,z,vx,vy,vz,jpos,jvel,jtot\n")
			for i in range(len(t)):
				J_pos = (np.sum((np.array([x[i],y[i]]) - pstar[0:2])**2))
				J_vel = vel_cost(np.array([x[i],y[i]]), pstar, np.array([vx[i],vy[i]]))
				J_tot = np.log(J_pos * (J_vel * gamma + 1))
				f.write(str(t[i]) + "," + \
						str(pstar[0]) + "," + \
						str(pstar[1]) + "," + \
						str(pstar[2]) + "," + \
						str(x[i]) + "," + \
						str(y[i]) + "," + \
						str(z[i]) + "," + \
						str(vx[i]) + "," + \
						str(vy[i]) + "," + \
						str(vz[i]) + "," + \
						str(J_pos) + "," + \
						str(J_vel) + "," + \
						str(J_tot) + \
						"\n")

		# Make a movie
		command = "ffmpeg" + " " + \
				"-r " +			"5" + " " + \
				"-f " +			"image2" + " " + \
				"-s " + 		"1920x1080" + " " + \
				"-i " + 		out_plot_folder + "%03d.png" + " " + \
				"-vcodec " +	"libx264" + " " + \
				"-crf " + 		"25" + " " + \
				movie_folder + movie_name + ".mp4"
		# os.system(command)