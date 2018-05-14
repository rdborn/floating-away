import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import neighbors

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, warning, compare, vector_sum, rng
from pyutils.constants import M_2_KM, DEG_2_RAD, KNT_2_MPS
from pyflow import flowfields

class Environment:
	def __init__(self, *args, **kwargs):
		type = parsekw(kwargs, 'environment', 'sounding')
		if type == 'sounding':
			self.wind = flowfields.SoundingField(**kwargs)
		elif type == 'sine':
			self.wind = flowfields.SineField(**kwargs)
		elif type == 'noaa':
			self.hr_curr = 0
			self.hr_inc = 3
			kwargs['hoursahead'] = self.hr_curr
			self.wind = flowfields.NOAAField(**kwargs)
			kwargs['hoursahead'] = self.hr_curr + self.hr_inc
			self.upcoming_wind = flowfields.NOAAField(**kwargs)
		elif type == 'brownsounding':
			self.wind = flowfields.BrownianSoundingField(**kwargs)
		elif type == 'brownsine':
			self.wind = flowfields.BrownianSineField(**kwargs)

	def get_flow(self, *args, **kwargs):
		p = parsekw(kwargs, 'p', np.inf * np.ones(3))
		t = parsekw(kwargs, 't', np.inf) / 3600.0
		if np.array(p == np.inf).any():
			print("No point specified, can't get flow")
			return False
		if t == np.inf:
			print("No time specified, can't get flow")
			return False
		if t > (self.hr_curr + self.hr_inc):
			self.hr_curr += self.hr_inc
			self.wind = flowfields.NOAAField(	origin=self.wind.origin,
												latspan=self.wind.lat_span_m,
												lonspan=self.wind.lon_span_m,
												hoursahead=self.hr_curr)
			self.upcoming_wind = flowfields.NOAAField(origin=self.upcoming_wind.origin,
												latspan=self.upcoming_wind.lat_span_m,
												lonspan=self.upcoming_wind.lon_span_m,
												hoursahead=(self.hr_curr + self.hr_inc))
		magnitude1, direction1 = self.wind.get_flow(p=p)
		magnitude2, direction2 = self.upcoming_wind.get_flow(p=p)
		vx1 = magnitude1 * np.cos(direction1)
		vy1 = magnitude1 * np.sin(direction1)
		vx2 = magnitude2 * np.cos(direction2)
		vy2 = magnitude2 * np.sin(direction2)
		mx = (vx2 - vx1) / self.hr_inc
		my = (vy2 - vy1) / self.hr_inc
		delta_t = t - self.hr_curr
		vx = vx1 + mx * delta_t
		vy = vy1 + my * delta_t
		return vx, vy
