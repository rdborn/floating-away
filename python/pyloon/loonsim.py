from pyflow.flowfields import SoundingField
from pyflow.flowfields import SineField
from pyloon.pyloon import GeneralLoon as Loon
from numpy import cos, sin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng

class LoonSim:
	def __init__(self, *args, **kwargs):
		"""
		Initialize balloon simulation.

		kwarg 'xi' initial x coordinate of the balloon (default 0.0).
		kwarg 'yi' initial y coordinate of the balloon (default 0.0).
		kwarg 'zi' initial z coordinate of the balloon (default average of zmin and zmax).
		kwarg 'Fs' sampling frequency (default 1.0 Hz)
		kwarg 'plot' boolean indicating whether to keep track of position history for plotting.

		If a sinusoidal field is desired:
		kwarg 'zmin' minimum altitude for which to generate a pyflow.flowfields.SineField.
		kwarg 'zmax' maximum altitude for which to generate a pyflow.flowfields.SineField.
		kwarg 'resolution' number of points at which to sample the pyflow.flowfields.SineField.
		kwarg 'frequency' frequency of the pyflow.flowfields.SineField
		kwarg 'amplitude' amplitude of the pyflow.flowfields.SineField
		kwarg 'phase' phase of the pyflow.flowfields.SineField
		kwarg 'offset' DC offset of the pyflow.flowfields.SineField

		If a field generated from sounding data is desired:
		kwarg 'file' file path to sounding data file
		"""

		self.tcurr = 0.0
		i_should_plot = parsekw(kwargs, 'plot', True)
		self.loon_history = DataFrame(columns=['t','x','y','z'])
		if i_should_plot:
			self.history_plot = plt.figure().gca(projection='3d')
			self.prev_plot_idx = 0
			plt.ion()

		file = parsekw(kwargs, 'file',"ERR_NO_FILE")
		if file == "ERR_NO_FILE":
			self.field = SineField( zmin=kwargs.get('zmin'),
									zmax=kwargs.get('zmax'),
									resolution=kwargs.get('resolution'),
									frequency=kwargs.get('frequency'),
									amplitude=kwargs.get('amplitude'),
									phase=kwargs.get('phase'),
									offset=kwargs.get('offset'))
		else:
			self.field = SoundingField(file=file)

		x = parsekw(kwargs, 'xi', 0.0)		# Initial x coordinate (m)  [default: (xdim/2) m]
		y = parsekw(kwargs, 'yi', 0.0)		# Initial x coordinate (m)  [default: (ydim/2) m]
		z = parsekw(kwargs, 'zi', (self.field.pmin[2] + self.field.pmax[2]) / 2.0)		# Initial x coordinate (m)  [default: (zdim/2) m]
		self.Fs = parsekw(kwargs, 'Fs', 1.0)					# sampling frequency (Hz)   [default: Fs = 1 Hz]
		self.loon = Loon(xi=x, yi=y, zi=z, Fs=self.Fs)
		self.dt = 1.0 / self.Fs
		loon_initial_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z]], columns=['t','x','y','z'])
		self.loon_history = self.loon_history.append(loon_initial_pos, ignore_index=True)

	def __str__(self):
		"""
		Return current balloon position and flow at that position.
		"""

		mag, angle = self.field.get_flow(self.loon.x, self.loon.y, self.loon.z)
		return "LOON POS: " + str(self.loon) + "\n" + "FLOW: mag: " + str(mag) + ", dir: " + str(angle) + "\n"

	def propogate(self, u):
		"""
		Propogate the simulation by one sampling period for a given control input.

		parameter u control effort (vertical velocity).
		"""
		# NOTE: terminal velocity is assumed at every time step
		#       (i.e. drag force reaches zero arbitrarily fast)

		magnitude, direction = self.field.get_flow(p=self.loon.get_pos())
		vloon = self.loon.get_vel()
		vx = magnitude * cos(direction)
		vy = magnitude * sin(direction)
		vz = u
		self.loon.update(vx=vx+rng(0), vy=vy, vz=vz+rng(0))
		self.tcurr += self.dt
		loon_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z]], columns=['t','x','y','z'])
		self.loon_history = self.loon_history.append(loon_pos, ignore_index=True)

	def __drag_force__(self, v):
		"""
		NOT SUPPORTED
		Calculate drag force for a given relative velocity.

		parameter v relative velocity for which to calculate drag force.
		return drag force on simulated balloon for relative velocity v.
		"""

		return v * abs(v) * self.field.density * self.loon.A * self.loon.Cd / 2

	def plot(self):
		"""
		Plot the balloon's position history since the last time this function was called.
		"""
		self.history_plot.scatter(	self.loon_history['x'][self.prev_plot_idx:],
									self.loon_history['y'][self.prev_plot_idx:],
									self.loon_history['z'][self.prev_plot_idx:])
		self.prev_plot_idx = len(self.loon_history['x']) - 1
		plt.pause(0.0001)

	def sample(self):
		"""
		NOT SUPPORTED
		Sample the wind velocity at the balloon's current position.
		"""
		p = self.loon.get_pos()
		magnitude, direction = self.field.get_flow(p=p)
		self.field.set_planar_flow(	z=p[2],
									magnitude=magnitude,
									direction=direction)
