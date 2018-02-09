import numpy as np

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, vector_sum
from pyutils.constants import M_2_KM, DEG_2_RAD, KNT_2_MPS
from skewt import SkewT

class FlowField:
	def __init__(self, *args, **kwargs):
		self.pmax = parsekw(kwargs, 'pmax', np.array([np.inf, np.inf, np.inf]))
		self.pmin = parsekw(kwargs, 'pmin', -np.array([np.inf, np.inf, np.inf]))
		self.density = parsekw(kwargs, 'density', 0.25)
		pass

	def __str__(self):
		return self.type

	def set_flow(self, *args, **kwargs):
		p = parsekw(kwargs, 'p', -np.inf)
		if any(np.array(p) == np.inf):
			return self.__warning__("No point specified. No action taken")
		if not self.__check_validity__(p):
			return self.__warning__("Invalid point. No action taken.")

	def get_flow(self, *args, **kwargs):
		pass

	def __check_validity__(self, p):
		valid = True
		valid &= p[0] <= self.pmax[0]
		valid &= p[1] <= self.pmax[1]
		valid &= p[2] <= self.pmax[2]
		valid &= p[0] >= self.pmin[0]
		valid &= p[1] >= self.pmin[1]
		valid &= p[2] >= self.pmin[2]
		return valid

	def __warning__(self, warning):
		print("WARNING: " + warning)
		return False


class PlanarField(FlowField):
	def __init__(self, *args, **kwargs):
		FlowField.__init__(	self,
							pmax=kwargs.get('pmax'),
							pmin=kwargs.get('pmin'),
							density=kwawrgs.get('density'))

	def set_flow(self, *args, **kwargs):
		if not FlowField.__check_validity__(self, p=kwargs.get('p')):
			return False
		self.magnitude[ x, y, z ] = mag		# magnitude in m/s
		self.direction[ x, y, z ] = angle	# angle in radians
		return True

	def set_planar_flow(self, z, mag, angle):
		if not FlowField.__check_validity__(self, 0, 0, z):
			return False
		for x in range(self.xdim):
			for y in range(self.ydim):
				self.set_flow(x, y, z, mag, angle)
		return True

	def get_flow(self, x, y, z):
		if not FlowField.__check_validity__(self, x, y, z):
			print("WARNING: requested coordinate outside flow field, returning 0 magnitude and 0 direction\n")
			return 0, 0

		# Only floorz and ceilz are used in the interpolation since
		# flow doesnt change with x and y in this particular sim
		interp_mag, interp_angle = self.__interp__(z)

		return interp_mag, interp_angle

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
