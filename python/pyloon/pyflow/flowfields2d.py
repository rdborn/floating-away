import numpy as np

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import vector_sum

class FlowField3DPlanar:
	xdim = 1
	ydim = 1
	zdim = 1
	dynamic_viscosity = 1
	density = 0.25

	def __init__(self, x, y, z):
		self.xdim = max(x, 1)
		self.ydim = max(y, 1)
		self.zdim = max(z, 1)
		self.magnitude = np.zeros((self.xdim, self.ydim, self.zdim))
		self.direction = np.zeros((self.xdim, self.ydim, self.zdim))

	def set_flow(self, x, y, z, mag, angle):
		if not self.__check_validity__(x, y, z):
			return False
		self.magnitude[ x, y, z ] = mag		# magnitude in m/s
		self.direction[ x, y, z ] = angle	# angle in radians
		return True

	def set_planar_flow(self, z, mag, angle):
		if not self.__check_validity__(0, 0, z):
			return False
		for x in range(self.xdim):
			for y in range(self.ydim):
				self.set_flow(x, y, z, mag, angle)
		return True

	def get_flow(self, x, y, z):
		if not self.__check_validity__(x, y, z):
			print("WARNING: requested coordinate outside flow field, returning 0 magnitude and 0 direction\n")
			return 0, 0

		# Only floorz and ceilz are used in the interpolation since
		# flow doesnt change with x and y in this particular sim
		interp_mag, interp_angle = self.__interp__(z)

		return interp_mag, interp_angle


	def __check_validity__(self, x, y, z):
		valid = True
		valid &= x < self.xdim
		valid &= y < self.ydim
		valid &= z < self.zdim
		valid &= x >= 0
		valid &= y >= 0
		valid &= z >= 0
		return valid

	def __interp__(self, z):
		z1 = int(np.ceil(z))
		z1mag = self.magnitude[0, 0, z1]
		z1angle = self.direction[0, 0, z1]
		if abs(z1 - z) < 1e-5:
			return z1mag, z1angle
		z2 = int(np.floor(z))
		z2mag = self.magnitude[0, 0, z2]
		z2angle = self.direction[0, 0, z2]
		dz = abs(z1 - z2)
		dz1 = abs(z - z1)
		dz2 = abs(z - z2)
		if dz == 0:
			print("WARNING: division by zero. Returning mag 0 dir 0")
		k1 = 1 - dz1 / dz
		k2 = 1 - dz2 / dz
		interp_mag, interp_angle = vector_sum(k1 * z1mag, z1angle, k2 * z2mag, z2angle)
		return interp_mag, interp_angle
