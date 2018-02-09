from pyflow.flowfields2d import FlowField3DPlanar as ff3
from pyflow.soundingfield import SoundingField as SF
from pyloon.multiinputloon import MultiInputLoon as Loon
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
		self.dt = 1.0
		self.tcurr = 0.0
		self.loon_history = DataFrame(columns=['t','x','y','z'])
		self.history_plot = plt.figure().gca(projection='3d')

		self.xdim = parsekw(kwargs.get('xdim'), 10.0)	# World x dimension (m)  [default: 10 m]
		self.ydim = parsekw(kwargs.get('ydim'), 10.0)	# World y dimension (m)  [default: 10 m]
		self.zdim = parsekw(kwargs.get('zdim'), 10.0)	# World z dimension (m)  [default: 10 m]
		x = parsekw(kwargs.get('xi'), self.xdim / 2.0)		# Initial x coordinate (m)  [default: (xdim/2) m]
		y = parsekw(kwargs.get('yi'), self.ydim / 2.0)		# Initial x coordinate (m)  [default: (ydim/2) m]
		z = parsekw(kwargs.get('zi'), self.zdim / 2.0)		# Initial x coordinate (m)  [default: (zdim/2) m]
		f = parsekw(kwargs.get('Fs'), 1.0)					# sampling frequency (Hz)   [default: Fs = 1 Hz]

		self.dt = 1.0 / f
		self.Fs = f
		file = parsekw(kwargs.get('file'),"ERR_NO_FILE")
		if file == "ERR_NO_FILE":
			self.field = ff3(self.xdim, self.ydim, self.zdim)
			self.loon = Loon(xi=x, yi=y, zi=z, Fs=f)
		else:
			self.field = SF(file=file)
			self.loon = Loon(xi=0.0, yi=0.0, zi=15000, Fs=f)
		loon_initial_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z]], columns=['t','x','y','z'])
		self.loon_history = self.loon_history.append(loon_initial_pos, ignore_index=True)

	def __str__(self):
		mag, angle = self.field.get_flow(self.loon.x, self.loon.y, self.loon.z)
		return "LOON POS: " + str(self.loon) + "\n" + "FLOW: mag: " + str(mag) + ", dir: " + str(angle) + "\n"

	# NOT SUPPORTED
	def set_sample_rate(self, hz):
		print("WARNING in set_sample_rate(): Changing the sampling rate after initialization may break the sim.")
		self.dt = 1.0 / hz

	def propogate(self, u):
		mag, angle = self.field.get_flow(self.loon.get_pos())
		vloon = self.loon.get_vel()
		fx = 0 #self.__drag_force__(mag * cos(angle) + rng(0.0) - vloon[0])
		fy = 0 #self.__drag_force__(mag * sin(angle) + rng(0.0) - vloon[1])
		vx = mag * cos(angle)
		vy = mag * sin(angle)
		vz = u
		self.loon.update(fx=fx, fy=fy, vx=vx, vy=vy, vz=vz)
		self.tcurr += self.dt
		loon_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z]], columns=['t','x','y','z'])
		self.loon_history = self.loon_history.append(loon_pos, ignore_index=True)

	def __drag_force__(self, v):
		return v * abs(v) * self.field.density * self.loon.A * self.loon.Cd / 2

	def plot(self):
		self.history_plot.scatter(self.loon_history['x'], self.loon_history['y'], self.loon_history['z'])
