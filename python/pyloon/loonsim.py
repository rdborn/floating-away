from pyflow.flowfields2d import FlowField3DPlanar as ff3
from pyloon.multiinputloon import MultiInputLoon as Loon
from numpy import cos, sin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D

class LoonSim:
	def __init__(self, *args, **kwargs):
		self.dt = 1.0
		self.tcurr = 0.0
		self.loon_history = DataFrame(columns=['t','x','y','z'])
		self.history_plot = plt.figure().gca(projection='3d')

		xdim = kwargs.get('xdim') != None
		ydim = kwargs.get('ydim') != None
		zdim = kwargs.get('zdim') != None
		xi = kwargs.get('xi') != None
		yi = kwargs.get('yi') != None
		zi = kwargs.get('zi') != None
		Fs = kwargs.get('Fs') != None

		self.xdim = kwargs.get('xdim') if xdim else 10.0	# World x dimension (m)  [default: 10 m]
		self.ydim = kwargs.get('ydim') if ydim else 10.0	# World y dimension (m)  [default: 10 m]
		self.zdim = kwargs.get('zdim') if zdim else 10.0	# World z dimension (m)  [default: 10 m]
		x = kwargs.get('xi') if xi else self.xdim / 2.0		# Initial x coordinate (m)  [default: (xdim/2) m]
		y = kwargs.get('yi') if yi else self.ydim / 2.0		# Initial x coordinate (m)  [default: (ydim/2) m]
		z = kwargs.get('zi') if zi else self.zdim / 2.0		# Initial x coordinate (m)  [default: (zdim/2) m]
		f = kwargs.get('Fs') if Fs else 1.0					# sampling frequency (Hz)   [default: Fs = 1 Hz]

		self.dt = 1.0 / f
		self.field = ff3(self.xdim, self.ydim, self.zdim)
		self.loon = Loon(xi=x, yi=y, zi=z, Fs=f)
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
		mag, angle = self.field.get_flow(self.loon.x, self.loon.y, self.loon.z)
		fd = self.__drag_force__(mag)
		fx = fd * cos(angle)
		fy = fd * sin(angle)
		vz = u
		self.loon.update(fx=fx, fy=fy, vz=vz)
		self.tcurr += self.dt
		loon_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z]], columns=['t','x','y','z'])
		self.loon_history = self.loon_history.append(loon_pos, ignore_index=True)

	def __drag_force__(self, v):
		return v * v * self.field.density * self.loon.A * self.loon.Cd / 2

	def plot(self):
		self.history_plot.scatter(self.loon_history['x'], self.loon_history['y'], self.loon_history['z'])
		plt.show()
