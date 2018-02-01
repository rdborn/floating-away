import numpy as np
import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))
from pybiquad.biquad import Integrator

class MultiInputLoon:
    # Init BiquadLoon
    def __init__(self, *args, **kwargs):
        self.Cd = 1	# drag coefficient
        self.A = 1	# cross-sectional area
        self.m = 1	# balloon mass

        # Set flags indicating which kwargs were provided
        xi = kwargs.get('xi') != None
        yi = kwargs.get('yi') != None
        zi = kwargs.get('zi') != None
        Fs = kwargs.get('Fs') != None
        m = kwargs.get('m') != None
        self.x = kwargs.get('xi') if xi else 0.0    # Initial x coordinate (m)  [default: 0 m]
        self.y = kwargs.get('yi') if yi else 0.0    # Initial y coordinate (m)  [default: 0 m]
        self.z = kwargs.get('zi') if zi else 0.0    # Initial z coordinate (m)  [default: 0 m]
        self.xi = self.x
        self.yi = self.y
        self.zi = self.z
        f = kwargs.get('Fs') if Fs else 1.0			# sampling frequency (Hz)   [default: Fs = 1 Hz]
        # Set up transfer function for motion in 3D
        self.Hx_a2v = Integrator(i=1.0, Fs=f)
        self.Hy_a2v = Integrator(i=1.0, Fs=f)
        self.Hz_a2v = Integrator(i=1.0, Fs=f)
        self.Hx_v2p = Integrator(i=1.0, Fs=f)
        self.Hy_v2p = Integrator(i=1.0, Fs=f)
        self.Hz_v2p = Integrator(i=1.0, Fs=f)

    def __str__(self):
        return "x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z)

    # update()
    # Propogates the balloon forward one time step under the provided inputs.
    # All inputs in a given axis at a given time step must be applied at once.
    def update(self, *args, **kwargs):
        # Set flags indicating which kwargs were provided
        fx = kwargs.get('fx') != None
        fy = kwargs.get('fy') != None
        fz = kwargs.get('fz') != None
        ax = kwargs.get('ax') != None
        ay = kwargs.get('ay') != None
        az = kwargs.get('az') != None
        vx = kwargs.get('vx') != None
        vy = kwargs.get('vy') != None
        vz = kwargs.get('vz') != None
        px = kwargs.get('px') != None
        py = kwargs.get('py') != None
        pz = kwargs.get('pz') != None
        f_x = kwargs.get('fx') if fx else 0.0
        f_y = kwargs.get('fy') if fy else 0.0
        f_z = kwargs.get('fz') if fz else 0.0
        a_x = kwargs.get('ax') if ax else 0.0
        a_y = kwargs.get('ay') if ay else 0.0
        a_z = kwargs.get('az') if az else 0.0
        v_x = kwargs.get('vx') if vx else 0.0
        v_y = kwargs.get('vy') if vy else 0.0
        v_z = kwargs.get('vz') if vz else 0.0
        p_x = kwargs.get('px') if px else 0.0
        p_y = kwargs.get('py') if py else 0.0
        p_z = kwargs.get('pz') if pz else 0.0
        self.__apply_accel__(ax=(a_x + f_x / self.m), ay=(a_y + f_y / self.m), az=(a_z + f_z / self.m))
        self.__apply_vel__(vx=v_x, vy=v_y, vz=v_z)
        self.__apply_pos__(px=p_x, py=p_y, pz=p_z)
        return self.get_pos()

    def __apply_accel__(self, *args, **kwargs):
        x = kwargs.get('ax') != None
        y = kwargs.get('ay') != None
        z = kwargs.get('az') != None
        ax = kwargs.get('ax') if x else 0.0
        ay = kwargs.get('ay') if y else 0.0
        az = kwargs.get('az') if z else 0.0
        self.Hx_a2v.update(ax)
        self.Hy_a2v.update(ay)
        self.Hz_a2v.update(az)

    def __apply_vel__(self, *args, **kwargs):
        x = kwargs.get('vx') != None
        y = kwargs.get('vy') != None
        z = kwargs.get('vz') != None
        vx = kwargs.get('vx') if x else 0.0
        vy = kwargs.get('vy') if y else 0.0
        vz = kwargs.get('vz') if z else 0.0
        self.Hx_v2p.update(vx + self.Hx_a2v.get_curr_val())
        self.Hy_v2p.update(vy + self.Hy_a2v.get_curr_val())
        self.Hz_v2p.update(vz + self.Hz_a2v.get_curr_val())

    def __apply_pos__(self, *args, **kwargs):
        x = kwargs.get('px') != None
        y = kwargs.get('py') != None
        z = kwargs.get('pz') != None
        px = kwargs.get('px') if x else 0.0
        py = kwargs.get('py') if y else 0.0
        pz = kwargs.get('pz') if z else 0.0
        self.x = self.xi + px + self.Hx_v2p.get_curr_val()
        self.y = self.yi + py + self.Hy_v2p.get_curr_val()
        self.z = self.zi + pz + self.Hz_v2p.get_curr_val()

    def get_pos(self):
        return self.x, self.y, self.z
