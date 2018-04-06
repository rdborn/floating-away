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
		self.tlast = 0.0
		self.resetting = False
		i_should_plot = parsekw(kwargs, 'plot', True)
		i_should_plot_to_file = parsekw(kwargs, 'plottofile', False)
		self.samplingtime = parsekw(kwargs, 'samplingtime', 0.)
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
                                        threshold=kwargs.get('threshold'),
										resamplethreshold=kwargs.get('resamplethreshold'),
										trusttime=kwargs.get('trusttime'))
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
			#	_________________   |	 |  |    |  |    |  |          |
			#	|                |  |	 |  |    |  |    |  |          |
			#	|________________|	|____|  |____|  |____|  |__________|
			#
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
			im = Image.open('stanford_area.png').convert('L') # grayscale
			# im = plt.imread('stanford_area.png') # color
			self.ax_latlon.imshow(im, extent=(-100,100,-100,100), cmap='gray')
			if not i_should_plot_to_file:
				plt.ion()
		self.loon_history = DataFrame(columns=['t','x','y','z','vx','vy','vz','stdx','stdy'])
		loon_initial_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z, 0.0, 0.0, 0.0, 0.0, 0.0]], columns=['t','x','y','z','vx','vy','vz','stdx','stdy'])
		self.loon_history = self.loon_history.append(loon_initial_pos, ignore_index=True)

	""" NOT SUPPORTED """
	def __str__(self):
		"""
		Return current balloon position and flow at that position.
		"""

		mag, angle = self.environment.wind.get_flow(self.loon.x, self.loon.y, self.loon.z)
		return "LOON POS: " + str(self.loon) + "\n" + "FLOW: mag: " + str(mag) + ", dir: " + str(angle) + "\n"

	def plan(self, *args, **kwargs):
		if (self.tcurr - self.tlast) / 3600.0 > self.environment.hr_inc:
			if self.pathplanner.planner.fieldestimator != 'naive':
				self.pathplanner.planner.retrain(field=self.environment.wind)
				self.tlast += (self.environment.hr_inc * 3600.0)
		return self.pathplanner.planner.plan(	loon=self.loon,
					                            u=kwargs.get('u'),
					                            T=kwargs.get('T'),
					                            pstar=kwargs.get('pstar'),
					                            depth=kwargs.get('depth'),
												tcurr=self.tcurr,
												gamma=kwargs.get('gamma'))
												
	def propogate(self, u, **kwargs):
		"""
		Propogate the simulation by one sampling period for a given control input.

		parameter u control effort (vertical velocity).
		"""
		# NOTE: terminal velocity is assumed at every time step
		#       (i.e. drag force reaches zero arbitrarily fast)

		dontsample = parsekw(kwargs, 'dontsample', False)
		alwayssample = parsekw(kwargs, 'alwayssample', False)

		p = self.loon.get_pos()
		vx, vy = self.environment.get_flow(p=p, t=self.tcurr)
		vx_pred, vy_pred, stdx, stdy = self.pathplanner.planner.predict(p)

		# Hacky way to push the agent back into the field
		# if it leaves the defined area
		magnitude = np.log(np.sqrt(p[0]**2 + p[1]**2) + 1)
		direction = np.arctan2(p[1], p[0]) + np.pi
		vx = vx if (abs(vx) > 0) else (magnitude * np.cos(direction))
		vy = vy if (abs(vy) > 0) else (magnitude * np.sin(direction))

		vz = u
		w = 1.0
		vloon = self.loon.get_vel()
		self.loon.update(vx=vx+rng(w), vy=vy+rng(w), vz=vz+rng(w))
		self.tcurr += self.dt

		loon_record = DataFrame([[self.tcurr,
								self.loon.x,
								self.loon.y,
								self.loon.z,
								vx,
								vy,
								vz,
								stdx[0],
								stdy[0]]],
								columns=['t','x','y','z','vx','vy','vz','stdx','stdy'])
		self.loon_history = self.loon_history.append(loon_record, ignore_index=True)

		p = self.loon.get_pos()
		i_want_to_sample = (abs(p[2] - self.pathplanner.planner.alts_to_sample) < 30)
		if i_want_to_sample.any():
			if dontsample:
				pass
			else:
				if alwayssample:
					print("Omg we're sampling")
					self.pathplanner.planner.alts_to_sample = self.pathplanner.planner.alts_to_sample[i_want_to_sample == False]
					self.sample()
					t_sample = self.tcurr
					while (self.tcurr - t_sample) < self.samplingtime:
						self.propogate(0, dontsample=True)
				else:
					pstar = np.zeros(3)
					vx_samp, vy_samp, stdx_samp, stdy_samp = self.pathplanner.planner.predict(p)
					going_the_right_way = self.pathplanner.planner.__cost_of_vel__(p[0:2], pstar,np.array([vx_samp, vy_samp])) < 1
					if going_the_right_way:
						print("Omg we're sampling")
						self.pathplanner.planner.alts_to_sample = self.pathplanner.planner.alts_to_sample[i_want_to_sample == False]
						self.sample()
						t_sample = self.tcurr
						while (self.tcurr - t_sample) < self.samplingtime:
							self.propogate(0, dontsample=True)

	def sample(self):
		"""
		Sample the wind velocity at the balloon's current position.
		"""
		p = self.loon.get_pos()
		magnitude, direction = self.environment.wind.get_flow(p=p)
		self.pathplanner.planner.add_sample(p=p,
									magnitude=magnitude+rng(1),
									direction=direction+rng(0.1))

	""" DEPRECATED """
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
		naive = parsekw(kwargs, 'naive', False)

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
		all_stdxs = np.array(self.loon_history['stdx'][:])
		all_stdys = np.array(self.loon_history['stdy'][:])
		all_stdxs= np.sqrt(np.cumsum(all_stdxs**2 * self.dt))
		all_stdys= np.sqrt(np.cumsum(all_stdys**2 * self.dt))
		all_dists = np.sqrt(all_lats**2 + all_lons**2)
		all_t = self.dt * np.array(range(len(all_dists)))
		avg_dists = np.cumsum(all_dists) / all_t
		avg_d_alt = np.cumsum(abs(all_alts[1:] - all_alts[:-1]))
		avg_d_alt = np.append(np.zeros(1), avg_d_alt) / all_t
		avg_stdxs = np.cumsum(all_stdxs**2) / all_t
		avg_stdys = np.cumsum(all_stdys**2) / all_t
		# Get rid of the NaN from dividing by initial time = 0
		avg_dists[0] = 0
		avg_d_alt[0] = 0
		avg_stdxs[0] = 0
		avg_stdys[0] = 0
		avg_stdxs = np.sqrt(avg_stdxs)
		avg_stdys = np.sqrt(avg_stdys)

		self.prev_plot_idx = len(self.loon_history['x']) - 1

		patches = []
		if naive:
			sampled_alts = self.pathplanner.planner.data[:,0]
			sampled_vxs = self.pathplanner.planner.data[:,1]
			sampled_vys = self.pathplanner.planner.data[:,2]
			sampled_mags = np.sqrt(sampled_vxs**2 + sampled_vys**2)
			sampled_dirs = np.arctan2(sampled_vys, sampled_vxs)
			for i in range(len(sampled_alts)):
				dlon = np.log(sampled_mags[i]) * np.sin(sampled_dirs[i])
				dlat = np.log(sampled_mags[i]) * np.cos(sampled_dirs[i])
				arr = Arrow(0, sampled_alts[i]*1e-3, dlon, dlat, 1e0, facecolor='k', alpha=0.3)
				patches.append(arr)
		else:
			# Plot jetstreams
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
		if naive:
			pass
		else:
			vlat_test, vlon_test, std_x, std_y = self.pathplanner.planner.predict(p_test.T)
			vlat_test = np.squeeze(vlat_test)
			vlon_test = np.squeeze(vlon_test)
			mag_test = np.sqrt(vlat_test**2 + vlon_test**2)
			vlatmin, vlatmax = pyutils.get_samesign_bounds(vlat_test, std_x)
			vlonmin, vlonmax = pyutils.get_samesign_bounds(vlon_test, std_y)
			mag_test_bound1 = np.sqrt(vlatmax**2 + vlonmax**2)
			mag_test_bound2 = np.sqrt(vlatmax**2 + vlonmin**2)
			mag_test_bound3 = np.sqrt(vlatmin**2 + vlonmax**2)
			mag_test_bound4 = np.sqrt(vlatmin**2 + vlonmin**2)
			mag_test_upper = np.max([mag_test_bound1,
									mag_test_bound2,
									mag_test_bound3,
									mag_test_bound4], axis=0)
			mag_test_lower = np.min([mag_test_bound1,
									mag_test_bound2,
									mag_test_bound3,
									mag_test_bound4], axis=0)

			dir_test = np.zeros(len(vlat_test))
			dir_lower = np.zeros(len(vlat_test))
			dir_upper = np.zeros(len(vlat_test))
			for i in range(len(vlat_test)):
				dir_lower[i], dir_test[i], dir_upper[i] = pyutils.get_angle_range(vlat_test[i], vlon_test[i], std_x[i], std_y[i])
			unbounded_idx = (abs(abs(dir_lower - dir_upper) - 360) < 1e-6)
			dir_test = dir_test % 360
			dir_lower = dir_lower % 360
			dir_upper = dir_upper % 360
			dir_lower[unbounded_idx] = -45.0
			dir_upper[unbounded_idx] = 405.0
			dir_lower[dir_lower > dir_test] -= 360
			dir_upper[dir_upper < dir_test] += 360

		vx_truth, vy_truth = self.environment.get_flow(p=p_test, t=self.tcurr)
		mag_truth = np.sqrt(vx_truth**2 + vy_truth**2)
		dir_truth = np.arctan2(vy_truth, vx_truth)
		dir_truth = pyutils.continuify_angles((dir_truth * 180.0 / np.pi) % 360)

		z_test *= 1e-3
		lats *= 1e-3
		lons *= 1e-3
		alts *= 1e-3

		# PLOT
		avg_ctrl_hist, = self.ax_cost.plot(all_t/3600.0, avg_d_alt*1e1, c=0.3*np.ones(3), linestyle='dashed')
		avg_dist_hist, = self.ax_cost.plot(all_t/3600.0, avg_dists*1e-2, c=0.6*np.ones(3))
		avg_stdx_hist, = self.ax_cost.plot(all_t/3600.0, avg_stdxs*1e-1, c='b', linestyle='dashed')
		avg_stdy_hist, = self.ax_cost.plot(all_t/3600.0, avg_stdys*1e-1, c='r', linestyle='dashed')
		all_stdx_hist, = self.ax_cost.plot(all_t/3600.0, all_stdxs*1e-2, c='b')
		all_stdy_hist, = self.ax_cost.plot(all_t/3600.0, all_stdys*1e-2, c='r')
		dist_hist, = self.ax_cost.plot(t/3600.0, dists*1e-3, c='k')
		if naive:
			dir_profile = self.ax_alt_dir.scatter((sampled_dirs * 180.0 / np.pi) % 360, sampled_alts*1e-3, c='k')
			mag_profile = self.ax_alt_mag.scatter(sampled_mags, sampled_alts*1e-3, c='k')
		else:
			dir_profile, = self.ax_alt_dir.plot(dir_test, z_test, c='k')
			dir_upper, = self.ax_alt_dir.plot(dir_upper, z_test, c='k', linestyle='dashed')
			dir_lower, = self.ax_alt_dir.plot(dir_lower, z_test, c='k', linestyle='dotted')
			mag_profile, = self.ax_alt_mag.plot(mag_test, z_test, c='k')
			mag_upper, = self.ax_alt_mag.plot(mag_test_upper, z_test, c='k', linestyle='dashed')
			mag_lower, = self.ax_alt_mag.plot(mag_test_lower, z_test, c='k', linestyle='dashed')
			self.pathplanner.planner.plot(ax=self.ax_plan)
		dir_profile_truth, = self.ax_alt_dir.plot(dir_truth % 360, z_test, c='grey')
		mag_profile_truth, = self.ax_alt_mag.plot(mag_truth, z_test, c='grey')
		self.ax_latlon.plot(lons, lats, c='grey')

		# LEGEND
		self.ax_cost.legend([dist_hist, avg_dist_hist, avg_ctrl_hist],
							['Dist from set point', 'Mean dist from set point', 'Mean vertical velocity'],
							loc=1)
		if naive:
			pass
		else:
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
		self.ax_alt_dir.set_xlim([-45, 405])
		self.ax_alt_mag.set_xlim([0,50])
		self.ax_cost.set_xlim([0, np.max(all_t)*1.1/3600.0])

		# YLIM
		self.ax_latlon.set_ylim([-100,100])
		self.ax_alt_mag.set_ylim([0,30])
		self.ax_cost.set_ylim([0, 100])

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
			# mng = plt.get_current_fig_manager()
			# mng.resize(*mng.window.maxsize())
			plt.pause(0.0001)
		else:
			self.fancy_plot.set_size_inches((18,10), forward=True)
			self.fancy_plot.savefig(out_file, bbox_inches='tight')

		if naive:
			pass
		else:
			dir_upper.remove()
			dir_lower.remove()
			mag_upper.remove()
			mag_lower.remove()
		dir_profile.remove()
		mag_profile.remove()
		rects.remove()
		prev_latlon.remove()
		prev_alt.remove()
		cards.remove()
		dir_profile_truth.remove()
		mag_profile_truth.remove()
		avg_dist_hist.remove()
		avg_ctrl_hist.remove()
		avg_stdx_hist.remove()
		all_stdx_hist.remove()
		all_stdy_hist.remove()
		avg_stdy_hist.remove()

		print('\tavg dist, avg ctrl, time:')
		print('\t\t' + str(avg_dists[-1]) + "\t" + str(avg_d_alt[-1]) + "\t" + str(all_t[-1]/3600.0))
