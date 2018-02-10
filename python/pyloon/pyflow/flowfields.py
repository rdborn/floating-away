import numpy as np
import pandas as pd

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, warning, compare, vector_sum
from pyutils.constants import M_2_KM, DEG_2_RAD, KNT_2_MPS
from skewt import SkewT

class FlowField:
	def __init__(self, *args, **kwargs):
		self.pmax = parsekw(kwargs, 'pmax', np.array([np.inf, np.inf, np.inf]))
		self.pmin = parsekw(kwargs, 'pmin', -np.array([np.inf, np.inf, np.inf]))
		self.density = parsekw(kwargs, 'density', 0.25)
		self.field = dict()
		self.coords = dict()
		pass

	def __str__(self):
		return self.type

	def set_flow(self, *args, **kwargs):
		p = parsekw(kwargs, 'p', np.inf)
		magnitude = parsekw(kwargs, 'magnitude', 0.0)
		direction = parsekw(kwargs, 'direction', 0.0)
		if any(np.array(p) == np.inf):
			return warning("No point specified. Cannot set flow. No action taken.")
		if not self.__check_validity__(p):
			return warning("Invalid point. No action taken.")
		self.field[hash3d(p)] = [magnitude, direction]
		self.coords[hash3d(p)] = p
		return True

	def get_flow(self, *args, **kwargs):
		p = parsekw(kwargs, 'p', np.inf)
		if any(np.array(p) == np.inf):
			return warning("No point specified. Cannot get flow.")
		if not self.__check_validity__(p):
			return warning("Invalid point. Cannot get flow.")
		if hash3d(p) not in field.keys():
			return warning("Flow not set. Cannot get flow.")
		return self.field[hash3d(p)]

	def __check_validity__(self, p):
		valid = True
		valid &= p[0] <= self.pmax[0]
		valid &= p[1] <= self.pmax[1]
		valid &= p[2] <= self.pmax[2]
		valid &= p[0] >= self.pmin[0]
		valid &= p[1] >= self.pmin[1]
		valid &= p[2] >= self.pmin[2]
		return valid

	def __find__(self, *args, **kwargs):
		p = parsekw(kwargs, 'p', np.inf)
		n = parsekw(kwargs, 'n', 1)
		if any(np.array(p) == np.inf):
			return warning("No point specified. Cannot get flow.")
		if not self.__check_validity__(p):
			return warning("Invalid point. Cannot get flow.")
		relative_p = np.subtract(np.array(p), np.array(self.coords.values()))
		distances = np.sum(np.abs(relative_p)**2, axis=-1)**(1./2)
		z = np.zeros([n,len(self.coords.values()[0])])
		n_smallest = pd.Series(distances).nsmallest(n)
		for i in range(n):
			z[i] = np.array(self.coords.values())[distances == n_smallest.iloc[i]]
		return z

class PlanarField(FlowField):
	def __init__(self, *args, **kwargs):
		zmax = parsekw(kwargs, 'zmax',np.inf)
		zmin = parsekw(kwargs, 'zmin',-np.inf)
		pmax = np.array([np.inf, np.inf, zmax])
		pmin = np.array([-np.inf, -np.inf, zmin])
		FlowField.__init__(	self,
							pmax=kwargs.get('pmax'),
							pmin=kwargs.get('pmin'),
							density=kwargs.get('density'))

	def set_planar_flow(self, *args, **kwargs):
		z = parsekw(kwargs, 'z', np.inf)
		if z == np.inf:
			return warning("No point specified. Cannot set flow.")
		p = np.array([0, 0, z])
		return FlowField.set_flow(	self,
									p=p,
									magnitude=kwargs.get('magnitude'),
									direction=kwargs.get('direction'))

	def get_flow(self, *args, **kwargs):
		p = parsekw(kwargs, 'p', np.inf)
		if any(compare(p, np.inf)):
			return warning("No point specified. Cannot get flow.")
		# Only floorz and ceilz are used in the interpolation since
		# flow doesnt change with x and y in this particular sim
		return self.__interp__(z=p[2])

	def __interp__(self, *args, **kwargs):
		z = parsekw(kwargs, 'z', np.inf)
		if z == np.inf:
			return warning("No point specified. Cannot set flow.")
		p = np.array([0, 0, z])
		[p1, p2] = FlowField.__find__(self, p=p, n=2)
		[p1mag, p1dir] = self.field[hash3d(p1)]
		[p2mag, p2dir] = self.field[hash3d(p2)]
		dp = np.linalg.norm(np.subtract(p1, p2))
		dp1 = np.linalg.norm(np.subtract(p1, p))
		dp2 = np.linalg.norm(np.subtract(p2, p))
		if dp1 < 1e-5:
			return p1mag, p1angle
		if dp2 < 1e-5:
			return p2mag, p2angle
		if dp == 0:
			print("WARNING: division by zero. Returning mag 0 dir 0")
		k1 = 1 - dp1 / dp
		k2 = 1 - dp2 / dp
		interp_mag, interp_angle = vector_sum(k1 * p1mag, p1dir, k2 * p2mag, p2dir)
		return interp_mag, interp_angle

class SoundingField(PlanarField):
	def __init__(self, *args, **kwargs):
		file = parsekw(kwargs, 'file', "ERR_NO_FILE")
		wind_sounding = SkewT.Sounding(file)
		wind = wind_sounding.soundingdata
		altitude = np.array(wind['hght'])
		pressure = np.array(wind['pres'])
		direction = np.array(wind['drct']) * DEG_2_RAD
		speed = np.array(wind['sknt']) * KNT_2_MPS
		temperature = np.array(wind['temp'])
		PlanarField.__init__(	self,
								zmax=np.max(altitude),
								zmin=np.min(altitude),)
		for i in range(len(altitude)):
			PlanarField.set_planar_flow(self,
										z=altitude[i],
										magnitude=speed[i],
										direction=direction[i])

class SineField(PlanarField):
	def __init__(self, *args, **kwargs):
		zmin=parsekw(kwargs, 'zmin', 0.0)
		zmax=parsekw(kwargs, 'zmax', 0.0)
		PlanarField.__init__(	self,
								zmin=zmin,
								zmax=zmax)
		resolution = parsekw(kwargs, 'resolution', np.inf)
		frequency = parsekw(kwargs, 'frequency', 0.0)
		amplitude = parsekw(kwargs, 'amplitude', 0.0)
		phase = parsekw(kwargs, 'phase', 0.0)
		offset = parsekw(kwargs, 'offset', 0.0)
		altitude = np.linspace(zmin, zmax, resolution)
		for i in range(len(altitude)):
			magnitude = amplitude * np.sin(frequency * altitude[i] + phase) + offset
			PlanarField.set_planar_flow(self,
										z=altitude[i],
										magnitude=magnitude,
										direction=0.0)
