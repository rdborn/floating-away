import numpy as np
import cProfile
import datetime
import os
from pandas import DataFrame
from matplotlib import pyplot as plt

from pysim.loonsim import LoonSim
from pyutils.pyutils import parsekw

#############################################
# SETUP PARAMETERS
#############################################

case = '2b'
# depths = [1, 2, 4, 8]
depths = [1, 2, 4]
# depths = [2, 4]
# depths = [4]
# parent_dir = './quadcosts/'
parent_dir = './datasets/case-' + case + '/'

persistently_bad = False
t_started_being_bad = 0.0
timenormalize = False

if case == '0a':
	planner = 'mpcfast'
	cost = 'Jv + Jp'
	restrict = False
	restriction = 1.
	timenormalize = False
elif case == '0b':
	planner = 'mpcfast'
	cost = 'Jv + Jp'
	restrict = True
	restriction = 0.5
	timenormalize = False
elif case == '0c':
	planner = 'mpcfast'
	cost = 'Jv + Jp'
	restrict = True
	restriction = 0.25
	timenormalize = False
elif case == '1a':
	planner = 'mpcfast'
	cost = 'Jv'
	restrict = False
	restriction = 1.
	timenormalize = False
elif case == '1b':
	planner = 'mpcfast'
	cost = 'Jv'
	restrict = True
	restriction = 0.5
	timenormalize = False
elif case == '1c':
	planner = 'mpcfast'
	cost = 'Jv'
	restrict = True
	restriction = 0.25
	timenormalize = False
elif case == '2a':
	planner = 'mpcfast'
	cost = 'Jp'
	restrict = False
	restriction = 1.
	timenormalize = False
elif case == '2b':
	planner = 'mpcfast'
	cost = 'Jp'
	restrict = True
	restriction = 0.5
	timenormalize = False
elif case == '2c':
	planner = 'mpcfast'
	cost = 'Jp'
	restrict = True
	restriction = 0.25
	timenormalize = False
elif case == '2d':
	planner = 'mpcfast'
	cost = 'Jp'
	restrict = False
	restriction = 1.
	timenormalize = True
elif case == '2e':
	planner = 'mpcfast'
	cost = 'Jp'
	restrict = True
	restriction = 0.5
	timenormalize = True
elif case == '2f':
	planner = 'mpcfast'
	cost = 'Jp'
	restrict = True
	restriction = 0.25
	timenormalize = True
elif case == '2g':
	planner = 'mpcfast'
	cost = 'Jp + Jd_f'
	restrict = False
	restriction = 1.
	timenormalize = False
elif case == '2h':
	planner = 'mpcfast'
	cost = 'Jp + Jd_f'
	restrict = True
	restriction = 0.5
	timenormalize = False
elif case == '2i':
	planner = 'mpcfast'
	cost = 'Jp + Jd_f'
	restrict = True
	restriction = 0.25
	timenormalize = False
elif case == '2j':
	planner = 'mpcfast'
	cost = 'Jp + Jd_f'
	restrict = False
	restriction = 1.
	timenormalize = True
elif case == '2k':
	planner = 'mpcfast'
	cost = 'Jp + Jd_f'
	restrict = True
	restriction = 0.5
	timenormalize = True
elif case == '2l':
	planner = 'mpcfast'
	cost = 'Jp + Jd_f'
	restrict = True
	restriction = 0.25
	timenormalize = True
elif case == '2m':
	planner = 'mpcfast'
	cost = 'Jp + Jd'
	restrict = False
	restriction = 1.
	timenormalize = False
elif case == '2n':
	planner = 'mpcfast'
	cost = 'Jp + Jd'
	restrict = False
	restriction = 0.5
	timenormalize = False
elif case == '2o':
	planner = 'mpcfast'
	cost = 'Jp + Jd'
	restrict = True
	restriction = 0.25
	timenormalize = False
elif case == '2p':
	planner = 'mpcfast'
	cost = 'Jp + Jd'
	restrict = False
	restriction = 1.
	timenormalize = True
elif case == '2q':
	planner = 'mpcfast'
	cost = 'Jp + Jd'
	restrict = True
	restriction = 0.5
	timenormalize = True
elif case == '2r':
	planner = 'mpcfast'
	cost = 'Jp + Jd'
	restrict = True
	restriction = 0.25
	timenormalize = True
elif case == '3a':
	planner = 'molchanov'
	restrict = False
	restriction = 1.
	timenormalize = False
elif case == '3b':
	planner = 'molchanov'
	restrict = True
	restriction = 0.5
	timenormalize = False
elif case == '3c':
	planner = 'molchanov'
	restrict = True
	restriction = 0.25
	timenormalize = False
elif case == '4a':
	planner = 'pic'
	restrict = False
	restriction = 1.
	timenormalize = False
elif case == '4b':
	planner = 'pic'
	restrict = True
	restriction = 0.5
	timenormalize = False
elif case == '4c':
	planner = 'pic'
	restrict = True
	restriction = 0.25
	timenormalize = False
elif case == '5a':
	planner = 'stayaloft'
	restrict = False
	restriction = 1.
	timenormalize = False
elif case == '5b':
	planner = 'stayaloft'
	restrict = False
	restriction = 1.
	timenormalize = False
elif case == '5c':
	planner = 'stayaloft'
	restrict = False
	restriction = 1.
	timenormalize = False



# Choose the planner algorithm and estimator scheme
# options: montecarlo, mpc, mpcfast, ldp, wap, pic, naive
# planner = 'mpcfast'
# planner = 'molchanov'
alwayssample = False
dontsample = True
samplingtime = 0.1*300
# gamma = np.array([1.,1e4,0.,0.,0.,0.,0.,0.]) # tuning parameter for cost function (lower = care more about pos, less about vel)
# gamma = np.array([1.,1e4,0.,0.,0.,0.,0.,0.]) # tuning parameter for cost function (lower = care more about pos, less about vel)
# gamma = np.array([1.,0.,0.,0.,0.,0.,0.,1e5]) # tuning parameter for cost function (lower = care more about pos, less about vel)
gamma = np.array([1.,0.,0.,0.,0.,0.,0.,0.]) # tuning parameter for cost function (lower = care more about pos, less about vel)
# gamma = np.array([0.,1.,0.,0.,0.,0.,0.,0.]) # tuning parameter for cost function (lower = care more about pos, less about vel)
while alwayssample and gamma[-1] != 0:
	print("WOAH you sure about that?")
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
resamplethreshold = 2000000
trusttime = 12

# Min and max altitude for which there is flow data
zmin = 0.0
zmax = 30000.0

# Choose parameters for identifying jetstreams in the wind field
streamres =		1000
streammin =		lower
streammax =		upper
M_PER_SAMPLE =	((streammax - streammin) / streamres)
streamsize = 	np.int(np.ceil(850 / M_PER_SAMPLE))
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

def choose_ctrl(LS, xstar, **kwargs):
	exact = parsekw(kwargs, 'exact', False)
	x = LS.loon.get_pos()[2]
	if not exact:
		x_id = LS.pathplanner.planner.jets.find(x).id
		xstar_id = LS.pathplanner.planner.jets.find(xstar).id
		if x_id == xstar_id:
			return 0
	if xstar - x > 0:
		return 1
	elif xstar - x < 0:
		return -1
	else:
		return 0

def get_there(LS, target, u, T, **kwargs):
	pos = LS.loon.get_pos()
	p = pos[2]
	c = choose_ctrl(LS, target, **kwargs)
	if c == 0:
		dt = 0.0
		while (T - dt) > 0.0:
			LS.propogate(c*u, alwayssample=alwayssample, dontsample=dontsample)
			dt += 1.0 / LS.loon.Fs
	else:
		while (target - p) * c * u > 0:
			LS.propogate(c*u, alwayssample=alwayssample, dontsample=dontsample)
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
	for depth in depths:
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
					trusttime=trusttime,
					samplingtime=samplingtime,
					restrict=restrict,
					restriction=restriction,
					timenormalize=timenormalize)

		# points_to_sample = np.sort(np.random.uniform(lower, upper, n_samples))
		movie_name = planner + '_' + \
					str(depth) + '-depth_' + \
					environment + '_' + \
					datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
		record_name = movie_name
		out_plot_folder = parent_dir + 'fig_' + movie_name + '/'
		movie_folder = out_plot_folder
		record_folder = out_plot_folder


		if plot:
			if plottofile:
				if not os.path.exists(out_plot_folder):
					os.mkdir(out_plot_folder)
				with open((out_plot_folder + "README.txt"), "w+") as f:
					f.write("SIMULATION PARAMETERS\n" + \
						"Planner:\t" + planner + "\n" + \
						"Don't sample:\t" + str(dontsample) + " (if True, overrides always sample)\n" + \
						"Always sample:\t" + str(alwayssample) + "(shouldn't be True with gamma != [X, X, X, X, X, 0])\n"\
						"Sampling time:\t" + str(samplingtime) + "\n" + \
						"Reduced data:\t0.5\n" + \
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
						"Number of Samples:\t" + "N/A" + "\n" +  \
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
					f.write("\tN/A\n")
				out_plot_folder = out_plot_folder + 'png/'
				if not os.path.exists(out_plot_folder):
					os.mkdir(out_plot_folder)
		pos = LS.loon.get_pos()
		buf = 30.0
		n_nodes_desired = 150
		it = 0
		n_frames = 200
		while LS.tcurr < 46 * 3600:
			u = 5.0
			# Set output file for plots
			out_file = out_plot_folder + str(it).zfill(len(str(n_frames))) + '.png'
			# Increment iterator
			it += 1
			print(str(it) + " Planning... depth: " + str(depth))
			# Figure out our next move
			pol = LS.plan(	u=u,
							T=T,
							pstar=pstar,
							depth=depth,
							gamma=gamma)
			# if plot:
			# 	if plottofile:
			# 		LS.plot(outfile=out_file, planner=planner)
			# 	else:
			# 		LS.plot(planner=planner)
			if planner == 'stayaloft':
				if pol < 0:
					u = 0.
					dt = 0.
					T = 180.
					while (T - dt) > 0.0:
						LS.propogate(u, alwayssample=alwayssample, dontsample=dontsample)
						dt += 1.0 / LS.loon.Fs
				else:
					u = 5.
					print pol
					get_there(LS, pol, u, T, exact=True)
			elif planner == 'naive':
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
						LS = get_there(LS, sample_alt, u, T)
						LS.sample()
					LS.pathplanner.planner.retrain()
				else:
					for i in range(N):
						if i >= len(pol):
							break
						print("Moving to altitude:")
						print("\t" + str(np.int(pol[i])))
						if LS.off_nominal():
							entropy_threshold = 2.
							if pol[1] > entropy_threshold:
								LS = get_there(LS, pol[i], u, T, exact=True)
								LS.sample()
							else:
								LS = get_there(LS, pol[i], u, T, exact=False)
						else:
							LS = get_there(LS, pol[i], u, T, exact=False)
			elif planner == 'pic':
				pos = LS.loon.get_pos()
				if LS.tcurr == 0.:
					get_there(LS, (lower+upper)/2, u, T, exact=True)
				elif pos[2] < lower:
					get_there(LS, lower, u, T, exact=True)
				elif pos[2] > upper:
					get_there(LS, upper, u, T, exact=True)
				else:
					u = pol
					dt = 0.
					T = 180.
					while (T - dt) > 0.0:
						LS.propogate(u, alwayssample=alwayssample, dontsample=dontsample)
						dt += 1.0 / LS.loon.Fs
			elif planner == 'mpcfast':
				if LS.off_nominal():
					print("Off-nominal Policy:")
					print("\t" + str(pol))
				else:
					pol = pol[1:] if len(pol) > 1 else pol
					print("Policy:")
					print("\t" + str(pol))
				for i in range(N):
					if i >= len(pol):
						break
					print("Moving to altitude:")
					print("\t" + str(np.int(pol[i])))
					if LS.off_nominal():
						entropy_threshold = 2.
						if pol[1] > entropy_threshold:
							LS = get_there(LS, pol[i], u, T, exact=True)
							LS.sample()
						else:
							LS = get_there(LS, pol[i], u, T, exact=False)
					else:
						LS = get_there(LS, pol[i], u, T, exact=False)
			elif planner == 'molchanov':
				print(pol)
				if pol[0] < 0:
					u = 0.
					dt = 0.
					T = 180.
					while (T - dt) > 0.0:
						LS.propogate(u, alwayssample=alwayssample, dontsample=dontsample)
						dt += 1.0 / LS.loon.Fs
				elif len(pol) > 1:
					for i in range(len(pol)):
						print("Moving to altitude:")
						print("\t" + str(np.int(pol[i])))
						LS = get_there(LS, pol[i], u, T, exact=True)
						LS.sample()
					LS.pathplanner.planner.retrain()
				else:
					LS = get_there(LS, pol[0], u, T, exact=True)

			pos = LS.loon.get_pos()
			print("New position:")
			print("\t(" + str(np.int(pos[0])) + ", " + str(np.int(pos[1])) + ", " + str(np.int(pos[2])) + ")")
			print("Simulation time:")
			print("\t" + str(LS.tcurr / 3600.))
			all_lats = np.array(LS.loon_history['x'][:])
			all_lons = np.array(LS.loon_history['y'][:])
			all_dists = np.sqrt(all_lats**2 + all_lons**2)
			all_t = LS.dt * np.array(range(len(all_dists)))
			# avg_dists = np.mean(all_dists) #np.cumsum(all_dists) / all_t
			# TODO: make a better stopping criteria
			# if avg_dists[-1] > 20000:
				# break
			if np.linalg.norm(pos[:2]) > 125000:
				if persistently_bad:
					if abs(LS.tcurr - t_started_being_bad) > 3600:
						break
				else:
					persistently_bad = True
					t_started_being_bad = LS.tcurr
			else:
				persistently_bad = False

		# Write history to csv file
		t = LS.loon_history['t'][:]
		x = LS.loon_history['x'][:]
		y = LS.loon_history['y'][:]
		z = LS.loon_history['z'][:]
		vx = LS.loon_history['vx'][:]
		vy = LS.loon_history['vy'][:]
		vx_pred = LS.loon_history['vx_pred'][:]
		vy_pred = LS.loon_history['vy_pred'][:]
		u = LS.loon_history['u'][:]
		stdx = LS.loon_history['stdx'][:]
		stdy = LS.loon_history['stdy'][:]
		off_nominal = LS.loon_history['off_nominal'][:]
		sampling = LS.loon_history['sampling'][:]
		with open((record_folder + record_name + ".csv"), "w+") as f:
			# f.write("jpos calculated as ||(x,y) - (xstar,ystar)||_2\n")
			# f.write("jvel calculated as ((((x,y) - (xstar,ystar))_hat . (vx,vy)_hat) + 1), where ()_hat is a normalizing operator\n")
			# f.write("jtot calculated as ln(jpos((jvel)(gamma)+1))\n")
			f.write("pstar = [0, 0]")
			f.write("BLANK")
			f.write("BLANK")
			f.write("t,x,y,z,u,vx,vy,vx_pred,vy_pred,stdx,stdy,off_nominal,sampling\n")
			for i in range(len(t)):
				# J_pos = (np.sum((np.array([x[i],y[i]]) - pstar[0:2])**2))
				# J_vel = vel_cost(np.array([x[i],y[i]]), pstar, np.array([vx[i],vy[i]]))
				# J_tot = np.log(J_pos * (J_vel * gamma + 1))
				f.write(str(t[i]) + "," + \
						str(x[i]) + "," + \
						str(y[i]) + "," + \
						str(z[i]) + "," + \
						str(u[i]) + "," + \
						str(vx[i]) + "," + \
						str(vy[i]) + "," + \
						str(vx_pred[i]) + "," + \
						str(vy_pred[i]) + "," + \
						str(stdx[i]) + "," + \
						str(stdy[i]) + "," + \
						str(off_nominal[i]) + "," + \
						str(sampling[i]) + \
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
