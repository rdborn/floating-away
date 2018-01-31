from pyflow.flowfields2d import FlowField3DPlanar as ff3
from pyloon.simpleloon import SimpleLoon
from numpy import cos, sin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D

class LoonSim:
	dt = 1.0
	tcurr = 0.0
	loon_history = DataFrame(columns=['t','x','y','z'])
	history_plot = plt.figure().gca(projection='3d')

	def __init__(self, x, y, z):
		self.field = ff3(x, y, z)
		self.loon = SimpleLoon(x/2, y/2, z/2)
		loon_initial_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z]], columns=['t','x','y','z'])
		self.loon_history = self.loon_history.append(loon_initial_pos, ignore_index=True)

	def __str__(self):
		mag, angle = self.field.get_flow(self.loon.x, self.loon.y, self.loon.z)
		return "LOON POS: " + str(self.loon) + "\n" + "FLOW: mag: " + str(mag) + ", dir: " + str(angle) + "\n"

	def set_sample_rate(self, hz):
		self.dt = 1.0 / hz

	def propogate(self, u):
		mag, angle = self.field.get_flow(self.loon.x, self.loon.y, self.loon.z)
		fd = self.__drag_force__(mag)
		dx = fd * cos(angle) * self.dt
		dy = fd * sin(angle) * self.dt
		dz = u * self.dt
		self.loon.travel(dx, dy, dz)
		self.tcurr += self.dt
		loon_pos = DataFrame([[self.tcurr, self.loon.x, self.loon.y, self.loon.z]], columns=['t','x','y','z'])
		self.loon_history = self.loon_history.append(loon_pos, ignore_index=True)

	def __drag_force__(self, v):
		return v * v * self.field.density * self.loon.A * self.loon.Cd / 2

	def plot(self):
		self.history_plot.scatter(self.loon_history['x'], self.loon_history['y'], self.loon_history['z'])
		plt.show()
