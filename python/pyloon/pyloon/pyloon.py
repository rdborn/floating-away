import numpy as np
import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))
from pybiquad.biquad import Integrator
from pyutils.pyutils import parsekw
import copy

class GeneralLoon:
    # Init BiquadLoon
    def __init__(self, *args, **kwargs):
        self.Cd = 0.5	# drag coefficient
        self.A = 10	# cross-sectional area
        self.m = parsekw(kwargs, 'm', 1.0)      # balloon mass
        self.Fs = parsekw(kwargs, 'Fs', 1.0)    # sampling frequency (Hz)   [default: Fs = 1 Hz]

        self.x = parsekw(kwargs, 'xi', 0.0)    # Initial x coordinate (m)  [default: 0 m]
        self.y = parsekw(kwargs, 'yi', 0.0)    # Initial y coordinate (m)  [default: 0 m]
        self.z = parsekw(kwargs, 'zi', 0.0)    # Initial z coordinate (m)  [default: 0 m]
        self.v_x = 0.0
        self.v_y = 0.0
        self.v_z = 0.0
        self.xi = self.x
        self.yi = self.y
        self.zi = self.z
        # Set up transfer function for motion in 3D
        self.Hx_a2v = Integrator(i=1.0, Fs=self.Fs)
        self.Hy_a2v = Integrator(i=1.0, Fs=self.Fs)
        self.Hz_a2v = Integrator(i=1.0, Fs=self.Fs)
        self.Hx_v2p = Integrator(i=1.0, Fs=self.Fs)
        self.Hy_v2p = Integrator(i=1.0, Fs=self.Fs)
        self.Hz_v2p = Integrator(i=1.0, Fs=self.Fs)

    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z)

    # update()
    # Propogates the balloon forward one time step under the provided inputs.
    # All inputs in a given axis at a given time step must be applied at once.
    def update(self, *args, **kwargs):
        f_x = parsekw(kwargs, 'fx', 0.0)
        f_y = parsekw(kwargs, 'fy', 0.0)
        f_z = parsekw(kwargs, 'fz', 0.0)
        a_x = parsekw(kwargs, 'ax', 0.0)
        a_y = parsekw(kwargs, 'ay', 0.0)
        a_z = parsekw(kwargs, 'az', 0.0)
        v_x = parsekw(kwargs, 'vx', 0.0)
        v_y = parsekw(kwargs, 'vy', 0.0)
        v_z = parsekw(kwargs, 'vz', 0.0)
        p_x = parsekw(kwargs, 'px', 0.0)
        p_y = parsekw(kwargs, 'py', 0.0)
        p_z = parsekw(kwargs, 'pz', 0.0)
        # print("fx: " + str(f_x) + ", fy: " + str(f_y) + ", vx: " + str(v_x) + ", vy: " + str(v_y) + ", vz: " + str(v_z))
        self.__apply_accel__(   ax=(a_x + f_x / self.m),
                                ay=(a_y + f_y / self.m),
                                az=(a_z + f_z / self.m))
        self.__apply_vel__(     vx=v_x, vy=v_y, vz=v_z)
        self.__apply_pos__(     px=p_x, py=p_y, pz=p_z)
        return self.get_pos()

    def __apply_accel__(self, *args, **kwargs):
        ax = parsekw(kwargs, 'ax', 0.0)
        ay = parsekw(kwargs, 'ay', 0.0)
        az = parsekw(kwargs, 'az', 0.0)
        self.Hx_a2v.update(ax)
        self.Hy_a2v.update(ay)
        self.Hz_a2v.update(az)

    def __apply_vel__(self, *args, **kwargs):
        vx = parsekw(kwargs, 'vx', 0.0)
        vy = parsekw(kwargs, 'vy', 0.0)
        vz = parsekw(kwargs, 'vz', 0.0)
        self.Hx_v2p.update(vx + self.Hx_a2v.get_curr_val())
        self.Hy_v2p.update(vy + self.Hy_a2v.get_curr_val())
        self.Hz_v2p.update(vz + self.Hz_a2v.get_curr_val())
        self.v_x = vx + self.Hx_a2v.get_curr_val()
        self.v_y = vy + self.Hy_a2v.get_curr_val()
        self.v_z = vz + self.Hz_a2v.get_curr_val()

    def __apply_pos__(self, *args, **kwargs):
        px = parsekw(kwargs, 'px', 0.0)
        py = parsekw(kwargs, 'py', 0.0)
        pz = parsekw(kwargs, 'pz', 0.0)
        self.x = self.xi + px + self.Hx_v2p.get_curr_val()
        self.y = self.yi + py + self.Hy_v2p.get_curr_val()
        self.z = self.zi + pz + self.Hz_v2p.get_curr_val()

    def get_pos(self):
        return self.x, self.y, self.z

    def get_vel(self):
        return self.v_x, self.v_y, self.v_z
