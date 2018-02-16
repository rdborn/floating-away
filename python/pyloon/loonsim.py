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

		x = parsekw(kwargs, 'xi', self.field.pmax[0] / 2.0)		# Initial x coordinate (m)  [default: (xdim/2) m]
		y = parsekw(kwargs, 'yi', self.field.pmax[1] / 2.0)		# Initial x coordinate (m)  [default: (ydim/2) m]
		z = parsekw(kwargs, 'zi', self.field.pmax[2] / 2.0)		# Initial x coordinate (m)  [default: (zdim/2) m]
		self.Fs = parsekw(kwargs, 'Fs', 1.0)					# sampling frequency (Hz)   [default: Fs = 1 Hz]
		self.loon = Loon(xi=x, yi=y, zi=z, Fs=self.Fs)
		self.dt = 1.0 / self.Fs
		loon_initial_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z]], columns=['t','x','y','z'])
		self.loon_history = self.loon_history.append(loon_initial_pos, ignore_index=True)

	def __str__(self):
		mag, angle = self.field.get_flow(self.loon.x, self.loon.y, self.loon.z)
		return "LOON POS: " + str(self.loon) + "\n" + "FLOW: mag: " + str(mag) + ", dir: " + str(angle) + "\n"

	def propogate(self, u):
		magnitude, direction = self.field.get_flow(p=self.loon.get_pos())
		vloon = self.loon.get_vel()
		fx = 0 #self.__drag_force__(magnitude * cos(direction) + rng(0.0) - vloon[0])
		fy = 0 #self.__drag_force__(magnitude * sin(direction) + rng(0.0) - vloon[1])
		vx = magnitude * cos(direction)
		vy = magnitude * sin(direction)
		vz = u
		print("vx: " + str(vx))
		print("vy: " + str(vy))
		self.loon.update(fx=fx, fy=fy, vx=vx+rng(0), vy=vy, vz=vz+rng(0))
		self.tcurr += self.dt
		loon_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z]], columns=['t','x','y','z'])
		self.loon_history = self.loon_history.append(loon_pos, ignore_index=True)

	def __drag_force__(self, v):
		return v * abs(v) * self.field.density * self.loon.A * self.loon.Cd / 2

	def plot(self):
		self.history_plot.scatter(	self.loon_history['x'][self.prev_plot_idx:],
									self.loon_history['y'][self.prev_plot_idx:],
									self.loon_history['z'][self.prev_plot_idx:])
		self.prev_plot_idx = len(self.loon_history['x']) - 1
		plt.pause(0.0001)

	def sample(self):
		p = self.loon.get_pos()
		magnitude, direction = self.field.get_flow(p=p)
		self.field.set_planar_flow(	z=p[2],
									magnitude=magnitude,
									direction=direction)
