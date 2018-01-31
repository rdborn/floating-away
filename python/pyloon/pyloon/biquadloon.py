import numpy as np
import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))
from pybiquad.biquad import DoubleIntegrator

class BiquadLoon:
	Cd = 1	# drag coefficient
	A = 1	# cross-sectional area
	m = 1	# balloon mass

	# Init BiquadLoon
	#
	def __init__(self, *args, **kwargs):
		# Set flags indicating which kwargs were provided
		xi = kwargs.get('xi') != None
		yi = kwargs.get('yi') != None
		zi = kwargs.get('zi') != None
		Fs = kwargs.get('Fs') != None
		m = kwargs.get('m') != None

		k = 1.0 / kwargs.get('m') if m else 1.0		# inverse mass scaling factor (1/kg)	[default: m = 1 kg]
		f = kwargs.get('Fs') if Fs else 1.0			# sampling frequency (Hz)				[default: Fs = 1 Hz]

		# Set up transfer function for motion in 3D
		# Each transfer function takes a force as input and outputs a position
		self.Hx = DoubleIntegrator(i=k, Fs=f)
		self.Hy = DoubleIntegrator(i=k, Fs=f)
		self.Hz = DoubleIntegrator(i=k, Fs=f)

	def __str__(self):
		return "x: " + str(self.Hx.get_curr_val()) + ", y: " + str(self.Hy.get_curr_val()) + ", z: " + str(self.Hz.get_curr_val())

	def update(self, *args, **kwargs):
		# Set flags indicating which kwargs were provided
		fx = kwargs.get('fx') != None
		fy = kwargs.get('fy') != None
		fz = kwargs.get('fz') != None
		mag = kwargs.get('mag') != None
		phi = kwargs.get('phi') != None
		theta = kwargs.get('theta') != None

		components = fx or fy or fz
		mag_and_dir = mag or phi or theta

		if components and mag_and_dir:
			print("WARNING in update(): multiple update methods requested, no action taken.\n")
			return False
		elif components:
			f_x = kwargs.get('fx') if fx else 0.0
			f_y = kwargs.get('fy') if fy else 0.0
			f_z = kwargs.get('fz') if fz else 0.0
		elif mag_and_dir:
			p = kwargs.get('phi') if phi else 0.0
			t = kwargs.get('theta') if theta else 0.0
			f_z = mag * np.sin(theta)
			f_y = mag * np.cos(theta) * np.sin(phi)
			f_x = mag * np.cos(theta) * np.cos(phi)
		else:
			f_x = 0.0
			f_y = 0.0
			f_z = 0.0

		self.Hx.update(f_x)
		self.Hy.update(f_y)
		self.Hz.update(f_z)

		return True

	def get_pos(self):
		return self.Hx.get_curr_val(), self.Hy.get_curr_val(), self.Hz.get_curr_val()
