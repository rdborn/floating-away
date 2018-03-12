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
		type = parsekw(kwargs, 'type', 'sounding')
		if type == 'sounding':
			self.wind = flowfields.SoundingField(file=kwargs.get('file'))
		elif type == 'sine':
			self.wind = flowfields.SineField(	zmin=kwargs.get('zmin'),
												zmax=kwargs.get('zmax'),
												resolution=kwargs.get('resolution'),
												frequency=kwargs.get('frequency'),
												amplitude=kwargs.get('amplitude'),
												phase=kwargs.get('phase'),
												offset=kwargs.get('offset'))
		elif type == 'noaa':
			self.wind = flowfields.NOAAField(	origin=kwargs.get('origin'),
												latspan=kwargs.get('latspan'),
												lonspan=kwargs.get('lonspan'))
		elif type == 'brownsounding':
			self.wind = flowfields.BrownianSoundingField(file=kwargs.get('file'))
		elif type == 'brownsine':
			self.wind = flowfields.BrownianSineField(zmin=kwargs.get('zmin'),
												zmax=kwargs.get('zmax'),
												resolution=kwargs.get('resolution'),
												frequency=kwargs.get('frequency'),
												amplitude=kwargs.get('amplitude'),
												phase=kwargs.get('phase'),
												offset=kwargs.get('offset'))
