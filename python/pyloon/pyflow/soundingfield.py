import numpy as np

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, vector_sum
from skewt import SkewT

M_2_KM = 1e-3
DEG_2_RAD = np.pi / 180.0
KNT_2_MPS = 0.51444

class SoundingField:
    def __init__(self, *args, **kwargs):
        self.density = 0.25
        file = parsekw(kwargs.get('file'), "ERR_NO_FILE")
        wind_sounding = SkewT.Sounding(file)
        wind = wind_sounding.soundingdata
        altitude = np.array(wind['hght'])
        pressure = np.array(wind['pres'])
        direction = np.array(wind['drct']) * DEG_2_RAD
        speed = np.array(wind['sknt']) * KNT_2_MPS
        temperature = np.array(wind['temp'])
        self.field = dict()
        for i in range(len(altitude)):
            self.field[altitude[i]] = [speed[i], direction[i]]

    def get_flow(self, p):
        z = p[2]
        interp_mag, interp_angle = self.__interp__(z)
        return interp_mag, interp_angle

    def __interp__(self, z):
        sorted_keys = np.sort(self.field.keys())
        never_got_there = True
        for i in range(len(sorted_keys)):
            if sorted_keys[i] > z:
                z2 = sorted_keys[i]
                z1 = sorted_keys[i-1]
                never_got_there = False
                break
        if never_got_there:
            z2 = sorted_keys[-1]
            z1 = sorted_keys[-2]
        z1mag = self.field[z1][0]
        z1angle = self.field[z2][1]
        if abs(z1 - z) < 1e-5:
            return z1mag, z1angle
        z2mag = self.field[z2][0]
        z2angle = self.field[z2][1]
        dz = abs(z1 - z2)
        dz1 = abs(z - z1)
        dz2 = abs(z - z2)
        if dz == 0:
            print("WARNING: division by zero. Returning mag 0 dir 0")
        k1 = 1 - dz1 / dz
        k2 = 1 - dz2 / dz
        interp_mag, interp_angle = vector_sum(k1 * z1mag, z1angle, k2 * z2mag, z2angle)
        return interp_mag, interp_angle
