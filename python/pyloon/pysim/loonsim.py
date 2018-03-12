import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib.patches import Rectangle, Arrow
from matplotlib.collections import PatchCollection

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng
from pathplanner import PathPlanner
from environment import Environment
from pyloon.pyloon import GeneralLoon as Loon

class LoonSim:
	def __init__(self, *args, **kwargs):
		self.Fs = parsekw(kwargs, 'Fs', 1.0)					# sampling frequency (Hz)   [default: Fs = 1 Hz]
		self.dt = 1.0 / self.Fs
		self.tcurr = 0.0
		i_should_plot = parsekw(kwargs, 'plot', True)
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
			# self.history_plot = plt.figure().gca(projection='3d')
			self.fancy_plot = plt.figure()
			self.fancy_plot.set_figheight(5.0)
			self.fancy_plot.set_figwidth(8.0)
			pad = 3
			gs = gridspec.GridSpec(5*pad, 8*pad)
			self.ax_latlon = self.fancy_plot.add_subplot(gs[:-1*pad,:-3*(pad+1)])
			self.ax_alt_jets = self.fancy_plot.add_subplot(gs[:-1*pad,-3*pad:-2*pad])
			self.ax_alt_dir = self.fancy_plot.add_subplot(gs[:-1*pad,-2*pad:-1*pad], sharey=self.ax_alt_jets)
			self.ax_alt_mag = self.fancy_plot.add_subplot(gs[:-1*pad,-1*pad:], sharey=self.ax_alt_jets)
			self.ax_cost = self.fancy_plot.add_subplot(gs[-1*(pad-1):,:])
			self.prev_plot_idx = 0
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

	def plot(self):
		lats = np.array(self.loon_history['x'][self.prev_plot_idx:])
		lons = np.array(self.loon_history['y'][self.prev_plot_idx:])
		alts = np.array(self.loon_history['z'][self.prev_plot_idx:])
		dists = np.sqrt(lats**2 + lons**2)
		t = self.dt * np.array(range(len(dists)))
		t -= np.max(t)
		t += self.tcurr

		self.prev_plot_idx = len(self.loon_history['x']) - 1

		self.ax_cost.plot(t, dists, c='k')

		# Plot jetstreams
		patches = []
		for jet in self.pathplanner.planner.jets.jetstreams.values():
			height = jet.max_alt - jet.min_alt
			bottom = jet.min_alt
			width = 2*1e5
			direction = jet.direction
			magnitude = 1e3 * np.log(jet.magnitude)
			dx = magnitude * np.cos(direction)
			dy = magnitude * np.sin(direction)
			rect = Rectangle(np.array([-1e5,bottom]), width, height, facecolor='k', alpha=0.3)
			arr = Arrow(0, jet.avg_alt, dx, dy, 1e3, facecolor='k', alpha=0.3)
			patches.append(rect)
			patches.append(arr)
		collection = PatchCollection(patches, alpha=0.3)
		rects = self.ax_alt_jets.add_collection(collection)
		plt.setp(self.ax_alt_jets.get_xticklabels(), visible=False)

		# Plot actual field
		N_test = 200
		p_test = np.zeros([N_test, 3])
		z_test = np.linspace(0.0, 30000.0, N_test)
		p_test[:,0] = lats[0] * np.ones(N_test)
		p_test[:,1] = lons[0] * np.ones(N_test)
		p_test[:,2] = z_test

		vx_test, vy_test = self.pathplanner.planner.ev(p_test.T)
		mag_test = np.sqrt(vx_test**2 + vy_test**2)
		dir_test = (np.arctan2(vy_test, vx_test) * 180.0 / np.pi) % 360

		mag_truth, dir_truth = self.environment.wind.get_flow(p=p_test)
		dir_truth = (dir_truth * 180.0 / np.pi) % 360

		dir_profile, = self.ax_alt_dir.plot(dir_test, z_test, c='k')
		mag_profile, = self.ax_alt_mag.plot(mag_test, z_test, c='k')

		dir_profile_truth, = self.ax_alt_dir.plot(dir_truth, z_test, c='grey')
		mag_profile_truth, = self.ax_alt_mag.plot(mag_truth, z_test, c='grey')

		plt.setp(self.ax_alt_dir.get_yticklabels(), visible=False)
		plt.setp(self.ax_alt_mag.get_yticklabels(), visible=False)

		self.ax_latlon.plot(lats, lons, c='grey')
		# self.ax_alt_jets.plot(np.zeros(len(alts)), alts, c='k')

		prev_latlon = self.ax_latlon.scatter(lats[-1], lons[-1], c='k')
		prev_alt = self.ax_alt_jets.scatter(0, alts[-1], c='k')

		self.ax_latlon.set_xlim([-100000,100000])
		self.ax_latlon.set_ylim([-100000,100000])
		self.ax_latlon.set_aspect('equal')

		self.ax_alt_jets.set_xlim([-6000,6000])
		self.ax_alt_jets.set_ylim([0,30000])
		self.ax_alt_jets.set_aspect('equal')

		self.ax_alt_dir.set_xlim([0,360])

		self.ax_alt_mag.set_xlim([0,50])

		# plt.tight_layout()

		plt.pause(0.0001)

		prev_latlon.remove()
		prev_alt.remove()
		rects.remove()
		dir_profile.remove()
		mag_profile.remove()
		dir_profile_truth.remove()
		mag_profile_truth.remove()

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
