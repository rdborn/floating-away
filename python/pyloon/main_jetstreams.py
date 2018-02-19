from loonsim import LoonSim
from optiloon.loonpathplanner import MonteCarloPlanner as MCP
from optiloon.loonpathplanner import PlantInvertingController as PIC
from optiloon.loonpathplanner import WindAwarePlanner as WAP
from pyflow.pystreams import VarThresholdIdentifier as JSI
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
from skewt import SkewT

plotting = False

# Set up simulation parameters
hz = 0.2
duration = 6

# Set up flow field
# file = "./weather-data/oak_2017_07_01_00z.txt"
# file = "./weather-data/oak_2018_02_08_00z.txt"
# LS = LoonSim(file=file, Fs=hz, xi=10000.0, yi=10000.0, zi=15000.0, plot=False)
LS = LoonSim(	zmin=0.0,
			 	zmax=30000.0,
				resolution=100,
				frequency=2.0*np.pi/2000.0,
				amplitude=30.0,
				phase=0.0,
				offset=0.0,
				Fs=hz, xi=10000, yi=10000, zi=20500,
                plot=False)

# Set point
pstar = [0.0, 0.0, 13000.0]

last_pos = LS.loon.get_pos()
pos = last_pos
LPP = WAP(  field=LS.field,
            lo=10000,
            hi=30000)

z = np.linspace(0,30000,500)
vx = np.zeros(len(z))
vy = np.zeros(len(z))
for i, alt in enumerate(z):
	fx, fy = LPP.ev(np.array([0, 0, alt]))
	magnitude = np.sqrt(fx**2 + fy**2)
	direction = np.arctan2(fy, fx)
	vx[i] = magnitude # magnitude * np.cos(direction)
	vy[i] = np.cos(direction) # magnitude * np.sin(direction)
vx = vx / np.max(np.abs(vx))
vy = vy / np.max(np.abs(vy))
z = z / np.max(np.abs(z))
data = np.array([vx, vy, z]).T
threshold = 0.01
streamsize = 20
jets = JSI(data=data, threshold=threshold, streamsize=streamsize)
jets.plot()
plt.show()

# threshold = 1.0
# while(True):
#     print(threshold)
#     jets = JSI(data=data, threshold=threshold)
#     jets.plot()
#     plt.show()
#     threshold /= 2.0
