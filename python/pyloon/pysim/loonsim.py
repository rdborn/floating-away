import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib.patches import Rectangle, Arrow
from matplotlib.collections import PatchCollection
from PIL import Image

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng
from pyutils import pyutils
from pathplanner import PathPlanner
from environment import Environment
from pyloon.pyloon import GeneralLoon as Loon

class LoonSim:
	def __init__(self, *args, **kwargs):
		self.Fs = parsekw(kwargs, 'Fs', 1.0)					# sampling frequency (Hz)   [default: Fs = 1 Hz]
		self.dt = 1.0 / self.Fs
		self.tcurr = 0.0
		i_should_plot = parsekw(kwargs, 'plot', True)
		i_should_plot_to_file = parsekw(kwargs, 'plottofile', False)
		print("Setting up environment...")
		self.environment = Environment(	type=kwargs.get('environment'),
										file=kwargs.get('file'),
										origin=kwargs.get('origin'),
										latspan=kwargs.get('latspan'),
										lonspan=kwargs.get('lonspan'),
										zmin=kwargs.get('zmin'),
										zmax=kwargs.get('zmax'),
										resolution=kwargs.get('resolution'),
										frequency=kwargs.get('frequency'),
										amplitude=kwargs.get('amplitude'),
										phase=kwargs.get('phase'),
										offset=kwargs.get('offset'))
		print("Setting up path planner...")
		self.pathplanner = PathPlanner(	planner=kwargs.get('planner'),
										field=self.environment.wind,
                                        fieldestimator=kwargs.get('fieldestimator'),
                                        lower=kwargs.get('lower'),
                                        upper=kwargs.get('upper'),
                                        streamres=kwargs.get('streamres'),
                                        streammax=kwargs.get('streammax'),
                                        streammin=kwargs.get('streammin'),
                                        streamsize=kwargs.get('streamsize'),
                                        threshold=kwargs.get('threshold'))
		print("Setting up agent...")
		self.loon = Loon(				xi=parsekw(kwargs, 'xi', 0.0),
										yi=parsekw(kwargs, 'yi', 0.0),
										zi=parsekw(kwargs, 'zi', (self.environment.wind.pmin[2] + self.environment.wind.pmax[2]) / 2.0),
										Fs=self.Fs)
		if i_should_plot:
			print("Setting up plot...")
			#	________________	______  ______  ______  ____________
			#	|				|	|	 |  |    |  |    |  |          |
			#	|				|	|	 |  |    |  |    |  |          |
			#	|				|	|	 |  |    |  |    |  |          |
			#	|				|	|	 |  |    |  |    |  |          |
			#	|				|	|	 |  |    |  |    |  |          |
			#	|_______________|	|	 |  |    |  |    |  |          |
			#	________________    |	 |  |    |  |    |  |          |
			#	|              |    |	 |  |    |  |    |  |          |
			#	|______________|	|____|  |____|  |____|  |__________|
			#
			# self.history_plot = plt.figure().gca(projection='3d')
			self.fancy_plot = plt.figure()
			self.fancy_plot.set_figheight(5.0)
			self.fancy_plot.set_figwidth(8.0)
			pad = 3
			gs = gridspec.GridSpec(5*pad, 10*pad)
			self.ax_latlon = self.fancy_plot.add_subplot(	gs[:4*pad-2,	:4*pad-2], aspect='equal')
			self.ax_alt_mag = self.fancy_plot.add_subplot(	gs[:,			4*pad:5*pad])
			self.ax_alt_dir = self.fancy_plot.add_subplot(	gs[:,			5*pad:6*pad], sharey=self.ax_alt_mag)
			self.ax_alt_jets = self.fancy_plot.add_subplot(	gs[:,			6*pad:7*pad], adjustable='datalim', aspect='equal', sharey=self.ax_alt_mag)
			self.ax_plan = self.fancy_plot.add_subplot(		gs[:,			7*pad:], sharey=self.ax_alt_mag)
			self.ax_cost = self.fancy_plot.add_subplot(		gs[4*pad-1:,	:4*pad-1])
			self.prev_plot_idx = 0
			im = Image.open('stanford_area.png').convert('L')
			# im = plt.imread('stanford_area.png')
			self.ax_latlon.imshow(im, extent=(-100,100,-100,100), cmap='gray')
			if not i_should_plot_to_file:
				plt.ion()
		self.loon_history = DataFrame(columns=['t','x','y','z'])
		loon_initial_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z]], columns=['t','x','y','z'])
		self.loon_history = self.loon_history.append(loon_initial_pos, ignore_index=True)

	def __str__(self):
		"""
		Return current balloon position and flow at that position.
		"""

		mag, angle = self.environment.wind.get_flow(self.loon.x, self.loon.y, self.loon.z)
		return "LOON POS: " + str(self.loon) + "\n" + "FLOW: mag: " + str(mag) + ", dir: " + str(angle) + "\n"

	def plan(self, *args, **kwargs):
		return self.pathplanner.planner.plan(	loon=self.loon,
					                            u=kwargs.get('u'),
					                            T=kwargs.get('T'),
					                            pstar=kwargs.get('pstar'),
					                            depth=kwargs.get('depth'))
	def propogate(self, u):
		"""
		Propogate the simulation by one sampling period for a given control input.

		parameter u control effort (vertical velocity).
		"""
		# NOTE: terminal velocity is assumed at every time step
		#       (i.e. drag force reaches zero arbitrarily fast)

		magnitude, direction = self.environment.wind.get_flow(p=self.loon.get_pos())
		vloon = self.loon.get_vel()
		vx = magnitude * np.cos(direction)
		vy = magnitude * np.sin(direction)
		vz = u
		w = 1.0
		self.loon.update(vx=vx+rng(w), vy=vy+rng(w), vz=vz+rng(w))
		self.tcurr += self.dt
		loon_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z]], columns=['t','x','y','z'])
		self.loon_history = self.loon_history.append(loon_pos, ignore_index=True)

	def __plot__(self):
		"""
		Plot the balloon's position history since the last time this function was called.
		"""
		self.history_plot.scatter(	self.loon_history['x'][self.prev_plot_idx:],
									self.loon_history['y'][self.prev_plot_idx:],
									self.loon_history['z'][self.prev_plot_idx:])
		self.prev_plot_idx = len(self.loon_history['x']) - 1
		plt.pause(0.0001)

	def plot(self, *args, **kwargs):
		out_file = parsekw(kwargs, 'outfile', 'ERR_NO_FILE')

		lats = np.array(self.loon_history['x'][self.prev_plot_idx:])
		lons = np.array(self.loon_history['y'][self.prev_plot_idx:])
		alts = np.array(self.loon_history['z'][self.prev_plot_idx:])
		dists = np.sqrt(lats**2 + lons**2)
		t = self.dt * np.array(range(len(dists)))
		t -= np.max(t)
		t += self.tcurr

		all_lats = np.array(self.loon_history['x'][:])
		all_lons = np.array(self.loon_history['y'][:])
		all_alts = np.array(self.loon_history['z'][:])
		all_dists = np.sqrt(all_lats**2 + all_lons**2)
		all_t = self.dt * np.array(range(len(all_dists)))
		avg_dists = np.cumsum(all_dists) / all_t
		avg_d_alt = np.cumsum(abs(all_alts[1:] - all_alts[:-1]))
		avg_d_alt = np.append(np.zeros(1), avg_d_alt) / all_t
		# Get rid of the NaN from dividing by initial time = 0
		avg_dists[0] = 0
		avg_d_alt[0] = 0

		self.prev_plot_idx = len(self.loon_history['x']) - 1

		# Plot jetstreams
		patches = []
		for jet in self.pathplanner.planner.jets.jetstreams.values():
			height = (jet.max_alt - jet.min_alt)*1e-3
			bottom = jet.min_alt*1e-3
			width = 2*1e2
			direction = jet.direction
			magnitude = 1e0 * np.log(jet.magnitude)
			dlon = magnitude * np.sin(direction)
			dlat = magnitude * np.cos(direction)
			rect = Rectangle(np.array([-1e2,bottom]), width, height, facecolor='k', alpha=0.3)
			arr = Arrow(0, jet.avg_alt*1e-3, dlon, dlat, 1e0, facecolor='k', alpha=0.3)
			patches.append(rect)
			patches.append(arr)
		collection = PatchCollection(patches, alpha=0.3, facecolor='k')
		rects = self.ax_alt_jets.add_collection(collection)
		plt.setp(self.ax_alt_jets.get_xticklabels(), visible=False)

		# plot cardinal directions
		cardinal_dirs = []
		left_sides = np.linspace(-180,540,9) - 22.5
		NS = np.linspace(-180,540,5) - 22.5
		height = 30
		width = 45
		bottom = 0
		for left in left_sides:
			alpha = 0.6 if (left in NS) else 0.3
			# alpha = 0.3
			rect = Rectangle(np.array([left, bottom]), width, height, facecolor='k', alpha=alpha)
			cardinal_dirs.append(rect)
			if left in NS:
				cardinal_dirs.append(rect)
		cardinal_collection = PatchCollection(cardinal_dirs, alpha=0.1, facecolor='k')
		cards = self.ax_alt_dir.add_collection(cardinal_collection)

		# Plot actual field
		N_test = 1000
		p_test = np.zeros([N_test, 3])
		z_test = np.linspace(0.0, 30000.0, N_test)
		p_test[:,0] = lats[0] * np.ones(N_test)
		p_test[:,1] = lons[0] * np.ones(N_test)
		p_test[:,2] = z_test

		vlat_test, vlon_test, std_x, std_y = self.pathplanner.planner.predict(p_test.T)
		vlat_test = np.squeeze(vlat_test)
		vlon_test = np.squeeze(vlon_test)
		mag_test = np.sqrt(vlat_test**2 + vlon_test**2)
		mag_test_bound1 = np.sqrt((vlat_test + std_x)**2 + (vlon_test + std_y)**2)
		mag_test_bound2 = np.sqrt((vlat_test - std_x)**2 + (vlon_test - std_y)**2)
		mag_test_bound3 = np.sqrt((vlat_test + std_x)**2 + (vlon_test - std_y)**2)
		mag_test_bound4 = np.sqrt((vlat_test - std_x)**2 + (vlon_test + std_y)**2)
		mag_test_upper = np.max([mag_test_bound1,
								mag_test_bound2,
								mag_test_bound3,
								mag_test_bound4], axis=0)
		mag_test_lower = np.min([mag_test_bound1,
								mag_test_bound2,
								mag_test_bound3,
								mag_test_bound4], axis=0)
		zero_mag = ((np.array(abs(vlat_test) < std_x, dtype=int) + \
					np.array(abs(vlon_test) < std_y, dtype=int)) == 2)
		mag_test_lower[zero_mag] = 0.0

		dir_test = np.zeros(len(vlat_test))
		dir_lower = np.zeros(len(vlat_test))
		dir_upper = np.zeros(len(vlat_test))
		for i in range(len(vlat_test)):
			dir_lower[i], dir_test[i], dir_upper[i] = pyutils.get_angle_range(vlat_test[i], vlon_test[i], std_x[i], std_y[i])
		dir_test = pyutils.continuify_angles(dir_test)
		dir_lower = pyutils.continuify_angles(dir_lower)
		dir_upper = pyutils.continuify_angles(dir_upper)

		mag_truth, dir_truth = self.environment.wind.get_flow(p=p_test)
		dir_truth = pyutils.continuify_angles((dir_truth * 180.0 / np.pi) % 360)

		z_test *= 1e-3
		lats *= 1e-3
		lons *= 1e-3
		alts *= 1e-3

		# PLOT
		avg_ctrl_hist, = self.ax_cost.plot(all_t/3600.0, avg_d_alt*1e1, c=0.3*np.ones(3), linestyle='dashed')
		avg_dist_hist, = self.ax_cost.plot(all_t/3600.0, avg_dists*1e-2, c=0.6*np.ones(3))
		dist_hist, = self.ax_cost.plot(t/3600.0, dists*1e-3, c='k')
		dir_profile, = self.ax_alt_dir.plot(dir_test, z_test, c='k')
		dir_upper, = self.ax_alt_dir.plot(dir_upper, z_test, c='k', linestyle='dashed')
		dir_lower, = self.ax_alt_dir.plot(dir_lower, z_test, c='k', linestyle='dashed')
		dir_profile_truth, = self.ax_alt_dir.plot(dir_truth, z_test, c='grey')
		mag_profile, = self.ax_alt_mag.plot(mag_test, z_test, c='k')
		mag_upper, = self.ax_alt_mag.plot(mag_test_upper, z_test, c='k', linestyle='dashed')
		mag_lower, = self.ax_alt_mag.plot(mag_test_lower, z_test, c='k', linestyle='dashed')
		mag_profile_truth, = self.ax_alt_mag.plot(mag_truth, z_test, c='grey')
		self.pathplanner.planner.plot(ax=self.ax_plan)
		self.ax_latlon.plot(lons, lats, c='grey')

		# LEGEND
		self.ax_cost.legend([dist_hist, avg_dist_hist, avg_ctrl_hist],
							['Dist from set point', 'Mean dist from set point', 'Mean vertical velocity'],
							loc=4)
		self.ax_alt_mag.legend([mag_profile, mag_upper, mag_profile_truth],
								['Mean', 'Std Dev', 'Truth'],
								loc=4)

		# SCATTER
		prev_latlon = self.ax_latlon.scatter(lons[-1], lats[-1], c='k', zorder=10)
		prev_alt = self.ax_alt_jets.scatter(0, alts[-1], c='k')

		# SETP
		plt.setp(self.ax_alt_jets.get_yticklabels(), visible=False)
		plt.setp(self.ax_alt_dir.get_yticklabels(), visible=False)
		plt.setp(self.ax_plan.get_yticklabels(), visible=False)

		# XLIM
		self.ax_latlon.set_xlim([-100,100])
		self.ax_alt_dir.set_xlim([-202.5,562.5])
		self.ax_alt_mag.set_xlim([0,50])
		self.ax_cost.set_xlim([0, np.max(all_t)*1.1/3600.0])

		# YLIM
		self.ax_latlon.set_ylim([-100,100])
		self.ax_alt_mag.set_ylim([0,30])
		self.ax_cost.set_ylim([0, 60])

		# XLABEL
		self.ax_latlon.set_xlabel('x position [km]')
		self.ax_alt_mag.set_xlabel('speed (m/s)')
		self.ax_alt_dir.set_xlabel('direction [deg]')
		self.ax_plan.set_xlabel('cost')
		self.ax_cost.set_xlabel('time [hr]')

		# YLABEL
		self.ax_latlon.set_ylabel('y position [km]')
		self.ax_alt_mag.set_ylabel('z position [km]')
		self.ax_cost.set_ylabel('distance [km]\nmean distance [10x km]\nmean vertical vel [10x m/s]')

		# TITLE
		self.ax_latlon.set_title('Lateral Position')
		self.ax_alt_mag.set_title('Wind Speed')
		self.ax_alt_dir.set_title('Wind Direction')
		self.ax_alt_jets.set_title('Jetstreams')
		self.ax_plan.set_title('Tree Search Results')
		self.ax_cost.set_title('Lateral Distance from Set Point/Mean Vertical Velocity')


		if out_file == "ERR_NO_FILE":
			mng = plt.get_current_fig_manager()
			mng.resize(*mng.window.maxsize())
			plt.pause(0.0001)
		else:
			self.fancy_plot.set_size_inches((18,10), forward=True)
			self.fancy_plot.savefig(out_file, bbox_inches='tight')

		prev_latlon.remove()
		prev_alt.remove()
		rects.remove()
		cards.remove()
		dir_profile.remove()
		dir_upper.remove()
		dir_lower.remove()
		mag_profile.remove()
		mag_upper.remove()
		mag_lower.remove()
		dir_profile_truth.remove()
		mag_profile_truth.remove()
		avg_dist_hist.remove()
		avg_ctrl_hist.remove()

		print('\tavg dist, avg ctrl, time:')
		print('\t\t' + str(avg_dists[-1]) + "\t" + str(avg_d_alt[-1]) + "\t" + str(all_t[-1]/3600.0))

	def sample(self):
		"""
		NOT SUPPORTED
		Sample the wind velocity at the balloon's current position.
		"""
		p = self.loon.get_pos()
		magnitude, direction = self.environment.wind.get_flow(p=p)
		self.environment.wind.set_planar_flow(	z=p[2],
									magnitude=magnitude,
									direction=direction)
