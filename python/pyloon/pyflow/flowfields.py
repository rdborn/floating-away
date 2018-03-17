import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import neighbors

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, warning, compare, vector_sum, rng
from pyutils.constants import M_2_KM, DEG_2_RAD, KNT_2_MPS
from skewt import SkewT
from pynoaa.databringer import fetch

KM_2_NAUTMI = 0.539957
NAUTMI_2_KM = 1.0 / KM_2_NAUTMI
DEGLAT_2_NAUTMI = 60.0
NAUTMI_2_DEGLAT = 1.0 / DEGLAT_2_NAUTMI
M_2_KM = 1e-3
KM_2_M = 1e3
M_2_DEGLAT = M_2_KM * KM_2_NAUTMI * NAUTMI_2_DEGLAT
DEGLAT_2_M = 1.0 / M_2_DEGLAT

class FlowField:
	def __init__(self, *args, **kwargs):
		"""
		Initialize general flow field object.

		kwarg 'pmax' 3D point, upper corner of 3D bounding box (default [inf, inf, inf]).
		kwarg 'pmin' 3D point, lower corner of 3D bounding box (default [-inf, -inf, -inf]).
		kwarg 'density' density of air in this flow field (default 0.25).
		"""

		self.pmax = parsekw(kwargs, 'pmax', np.array([np.inf, np.inf, np.inf]))
		self.pmin = parsekw(kwargs, 'pmin', -np.array([np.inf, np.inf, np.inf]))
		self.density = parsekw(kwargs, 'density', 0.25)
		self.field = dict()
		self.coords = dict()

	def __str__(self):
		"""
		NOT SUPPORTED
		Return type designation of flow field.
		"""

		return self.type

	def set_flow(self, *args, **kwargs):
		"""
		Set the flow at a particular spot in the flow field.

		kwarg 'p' 3D point at which to set the flow.
		kwarg 'magnitude' speed of flow at p
		kwarg 'direction' direction of flow at p
		return success/failure
		"""

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
		"""
		Get the flow at a particular point in the flow field.

		kwarg 'p' point at which to get the flow.
		(if success) return magnitude and direction of flow at p
		(if failure) return failure
		"""

		p = parsekw(kwargs, 'p', np.inf)
		if any(np.array(p) == np.inf):
			return warning("No point specified. Cannot get flow.")
		if not self.__check_validity__(p):
			return warning("Invalid point. Cannot get flow.")
		if hash3d(p) not in self.field.keys():
			return warning("Flow not set. Cannot get flow.")
		return self.field[hash3d(p)]

	def __check_validity__(self, p):
		"""
		Check whether a point is in the flow field.

		parameter p 3D point to query.
		return valid/not valid
		"""

		valid = True
		valid &= p[0] <= self.pmax[0]
		valid &= p[1] <= self.pmax[1]
		valid &= p[2] <= self.pmax[2]
		valid &= p[0] >= self.pmin[0]
		valid &= p[1] >= self.pmin[1]
		valid &= p[2] >= self.pmin[2]
		return valid

	def __find__(self, *args, **kwargs):
		"""
		Find the nearest points with explicit values above and below in the flow field.

		kwarg p point at which to find the nearest point above and nearest point below.
		return nearest point above p and nearest point below p
		"""

		p = parsekw(kwargs, 'p', np.inf)
		if any(np.array(p) == np.inf):
			return warning("No point specified. Cannot get flow.")
		if not self.__check_validity__(p):
			return warning("Invalid point. Cannot get flow.")
		relative_p = np.subtract(np.array(p), np.array(self.coords.values()))
		relative_z = relative_p[:,2]
		idx_zero = (relative_z == 0)
		idx_neg = (relative_z < 0)
		idx_pos = (relative_z > 0)
		if idx_zero.any():
			z = np.array([	np.array(self.coords.values())[idx_zero],
							np.array(self.coords.values())[idx_zero]])
		else:
			distances = np.sum(np.abs(relative_p)**2, axis=-1)**(1./2)
			idx_next = (distances*idx_neg == np.min(distances[idx_neg]))
			idx_prev = (distances*idx_pos == np.min(distances[idx_pos]))
			if not idx_prev.any():
				idx_prev = idx_next
			if not idx_next.any():
				idx_next = idx_prev
			z = np.array([	np.array(self.coords.values())[idx_prev],
							np.array(self.coords.values())[idx_next]])
		return np.squeeze(z)

class NOAAField:
	def __init__(self, *args, **kwargs):
		self.origin = parsekw(kwargs, 'origin', np.array([37.0, -121.0]))
		self.lat_span_m = parsekw(kwargs, 'latspan', 120.0 * KM_2_M)
		self.lon_span_m = parsekw(kwargs, 'lonspan', 120.0 * KM_2_M)
		self.min_lat = self.origin[0] - self.lat_span_m * M_2_DEGLAT
		self.max_lat = self.origin[0] + self.lat_span_m * M_2_DEGLAT
		self.min_lon = self.origin[1] - self.lon_span_m * M_2_DEGLAT
		self.max_lon = self.origin[1] + self.lon_span_m * M_2_DEGLAT
		self.min_lon = self.min_lon if self.min_lon > 0 else self.min_lon + 180.0
		self.max_lon = self.max_lon if self.max_lon > 0 else self.max_lon + 180.0
		self.origin[1] = self.origin[1] if self.origin[1] > 0 else self.origin[1] + 180.0
		self.data = fetch(self.min_lat, self.min_lon, self.max_lat, self.max_lon, 15)
		self.field = dict()
		self.coords = dict()
		for d in self.data:
			p = d[0:3]
			vnorth = d[3]
			veast = d[4]
			magnitude = np.sqrt(vnorth**2 + veast**2)
			direction = np.arctan2(veast, vnorth)
			p[0] = (p[0] - self.origin[0]) * DEGLAT_2_M
			p[1] = (p[1] - self.origin[1]) * DEGLAT_2_M
			self.coords[hash3d(p)] = p
			self.field[hash3d(p)] = [magnitude, direction]
		n_neighbors = 4
		self.knn_vnorth = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
		self.knn_veast = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
		X = self.data[:,0:3]
		Ynorth = self.data[:,3]
		Yeast = self.data[:,4]
		self.knn_vnorth.fit(X, Ynorth)
		self.knn_veast.fit(X, Yeast)
		self.pmin = np.array([-self.lat_span_m, -self.lon_span_m, np.min(self.data[:,2])])
		self.pmax = np.array([self.lat_span_m, self.lon_span_m, np.max(self.data[:,2])])

	def get_flow(self, *args, **kwargs):
		"""
		Get flow at a given point.

		kwarg 'p' 3D point at which to get flow.
		return magnitude and direction of flow at p
		"""

		p = np.array(parsekw(kwargs, 'p', np.inf))
		if (compare(p, np.inf)).any():
			return warning("No point specified. Cannot get flow.")
		# p[0] = self.origin[0] + p[0] * M_2_KM * KM_2_NAUTMI * NAUTMI_2_DEGLAT
		# p[0] = self.origin[0] + p[0] * M_2_KM * KM_2_NAUTMI * NAUTMI_2_DEGLAT
		# p[1] = self.origin[1] + p[1] * M_2_KM * KM_2_NAUTMI * NAUTMI_2_DEGLAT
		vnorth = self.knn_vnorth.predict(np.atleast_2d(p))
		veast = self.knn_veast.predict(np.atleast_2d(p))
		magnitude = np.sqrt(vnorth**2 + veast**2)
		direction = np.arctan2(veast, vnorth)
		return [magnitude, direction]

class PlanarField(FlowField):
	def __init__(self, *args, **kwargs):
		"""
		Initialize flowfield whose value does not vary with x and y.

		kwarg 'zmin' minimum altitude at which to generate flow field.
		kwarg 'zmax' maximum altitude at which to generate flow field.
		kwarg 'density' density of air in this flow field.
		"""

		zmax = parsekw(kwargs, 'zmax',np.inf)
		zmin = parsekw(kwargs, 'zmin',-np.inf)
		pmax = np.array([np.inf, np.inf, zmax])
		pmin = np.array([-np.inf, -np.inf, zmin])
		FlowField.__init__(	self,
							pmax=pmax,
							pmin=pmin,
							density=kwargs.get('density'))

	def set_planar_flow(self, *args, **kwargs):
		"""
		Set flow at a given altitude.

		kwarg 'z' altitude at which to set flow.
		kwarg 'magnitude' speed of flow at z
		kwarg 'direction' direction of flow at z
		return success/failure
		"""

		z = parsekw(kwargs, 'z', np.inf)
		if z == np.inf:
			return warning("No point specified. Cannot set flow.")
		p = np.array([0, 0, z])
		return FlowField.set_flow(	self,
									p=p,
									magnitude=kwargs.get('magnitude'),
									direction=kwargs.get('direction'))

	def get_flow(self, *args, **kwargs):
		"""
		Get flow at a given point.

		kwarg 'p' 3D point at which to get flow.
		return magnitude and direction of flow at p
		"""

		p = parsekw(kwargs, 'p', np.inf)
		if (compare(p, np.inf)).any():
			return warning("No point specified. Cannot get flow.")
		return self.__interp__(z=p[2])

	def __interp__(self, *args, **kwargs):
		"""
		Interpolate between points with explicit values to get field value at a given point.

		kwarg 'z' altitude at which to find field value.
		return interpolated wind speed and interpolated direction
		"""

		z = parsekw(kwargs, 'z', np.inf)
		if z == np.inf:
			return warning("No point specified. Cannot set flow.")
		p = np.array([0, 0, z])
		[p1, p2] = FlowField.__find__(self, p=p)
		[p1mag, p1dir] = self.field[hash3d(p1)]
		[p2mag, p2dir] = self.field[hash3d(p2)]
		dp = np.linalg.norm(np.subtract(p1, p2))
		dp1 = np.linalg.norm(np.subtract(p1, p))
		dp2 = np.linalg.norm(np.subtract(p2, p))
		if dp1 < 1e-5:
			return p1mag, p1dir
		if dp2 < 1e-5:
			return p2mag, p2dir
		if dp < 1e-5:
			warning("Requested value outside bounds of reliable estimate.")
			return p1mag, p1dir
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
		X = np.zeros([len(self.field.keys()), 3])
		y_x = np.zeros(len(self.field.keys()))
		y_y = np.zeros(len(self.field.keys()))
		for i, key in enumerate(self.field.keys()):
			X[i] = self.coords[key]
			magnitude, direction = self.field[key]
			y_x[i] = magnitude * np.cos(direction)
			y_y[i] = magnitude * np.sin(direction)
		n_neighbors = 4
		self.knn_vx = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
		self.knn_vy = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
		self.knn_vx.fit(X, y_x)
		self.knn_vy.fit(X, y_y)

	def get_flow(self, *args, **kwargs):
		p = parsekw(kwargs, 'p', np.inf)
		if (compare(p, np.inf)).any():
			return warning("No point specified. Cannot get flow.")
		vx = self.knn_vx.predict(np.atleast_2d(p))
		vy = self.knn_vy.predict(np.atleast_2d(p))
		magnitude = np.sqrt(vx**2 + vy**2)
		direction = np.arctan2(vy, vx)
		return [magnitude, direction]

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

class BrownianSoundingField(SoundingField):
	def __init__(self, *args, **kwargs):
		SoundingField.__init__(self, file=kwargs.get('file'))
		self.brown_mag = np.zeros(len(self.coords.values()))
		self.brown_dir = np.zeros(len(self.coords.values()))

	def get_flow(self, *args, **kwargs):
		p = parsekw(kwargs, 'p', np.inf * np.ones(3))
		for i, pos in enumerate(self.coords.values()):
			curr_mag, curr_dir = FlowField.get_flow(self, p=pos)
			magnitude = curr_mag + self.brown_mag[i]
			direction = curr_dir + self.brown_dir[i]
			PlanarField.set_planar_flow(self, z=pos[2], magnitude=magnitude, direction=direction)
		self.brown_mag += norm.rvs(size=self.brown_mag.shape, scale=1e-3)
		self.brown_dir += norm.rvs(size=self.brown_dir.shape, scale=1e-6)
		return PlanarField.get_flow(self, p=p)

class BrownianSineField(SineField):
	def __init__(self, *args, **kwargs):
		SineField.__init__(	self,
							zmin=kwargs.get('zmin'),
							zmax=kwargs.get('zmax'),
							resolution=kwargs.get('resolution'),
							frequency=kwargs.get('frequency'),
							amplitude=kwargs.get('amplitude'),
							phase=kwargs.get('phase'),
							offset=kwargs.get('offset'))
		self.brown_mag = np.zeros(len(self.coords.values()))
		self.brown_dir = np.zeros(len(self.coords.values()))

	def get_flow(self, *args, **kwargs):
		p = parsekw(kwargs, 'p', np.inf * np.ones(3))
		for i, pos in enumerate(self.coords.values()):
			curr_mag, curr_dir = FlowField.get_flow(self, p=pos)
			magnitude = curr_mag + self.brown_mag[i]
			direction = curr_dir + self.brown_dir[i]
			PlanarField.set_planar_flow(self, z=pos[2], magnitude=magnitude, direction=direction)
		r_mag = norm.rvs(size=self.brown_mag.shape, scale=1e-3)
		r_dir = norm.rvs(size=self.brown_dir.shape, scale=1e-8)
		m = 20
		n = len(self.brown_mag)
		M = np.eye(n)
		for i in range(1,m):
			M += np.diag(np.ones(n-i),i)
			M += np.diag(np.ones(n-i),-i)
		M = np.matrix(M)
		self.brown_mag = np.matrix(self.brown_mag)
		self.brown_dir = np.matrix(self.brown_dir)
		r_mag = np.matrix(r_mag) * M / (2 * m)
		r_dir = np.matrix(r_dir) * M / (2 * m)
		self.brown_mag = np.squeeze(np.array(self.brown_mag + r_mag))
		self.brown_dir = np.squeeze(np.array(self.brown_dir + r_dir))
		return PlanarField.get_flow(self, p=p)
