from loonsim import LoonSim
from optiloon.optiloon import LoonPathPlanner
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt

# Set up simulation parameters
xdim = 10
ydim = 10
zdim = 10
hz = 20
duration = 6

# Set up flow field
LS = LoonSim(xdim=xdim, ydim=ydim, zdim=zdim, Fs=hz)
LS.field.set_planar_flow(int(np.ceil(0.0)),3.0,0.0)
LS.field.set_planar_flow(int(np.ceil(1.0)),7.0,0.0)
LS.field.set_planar_flow(int(np.ceil(2.0)),12.0,0.0)
LS.field.set_planar_flow(int(np.ceil(3.0)),16.0,0.0)
LS.field.set_planar_flow(int(np.ceil(4.0)),20.0,0.0)
LS.field.set_planar_flow(int(np.ceil(5.0)),15.0,0.0)
LS.field.set_planar_flow(int(np.ceil(6.0)),17.0,0.0)
LS.field.set_planar_flow(int(np.ceil(7.0)),11.0,0.0)
LS.field.set_planar_flow(int(np.ceil(8.0)),5.0,0.0)
LS.field.set_planar_flow(int(np.ceil(9.0)),0.0,0.0)

LPP = LoonPathPlanner(field=LS.field, res=1.2)

z = np.linspace(0,zdim-1,100)
fx = np.zeros(len(z))
fx_actual = np.zeros(len(z))
fx_std = np.zeros(len(z))
for i in range(len(z)):
	result = LPP.predict([3.5,3.5,z[i]])
	mag, angle = LS.field.get_flow(5,5,z[i])
	fx[i] = result[0]
	fx_actual[i] = mag * np.cos(angle)
	fx_std[i] = result[3]

fig = plt.figure()
plt.plot(z, fx_actual, 'r:')
plt.plot(z, fx, 'b-')
plt.fill(np.concatenate([z,z[::-1]]), np.concatenate([fx - 3*fx_std, (fx + 3*fx_std)[::-1]]), alpha=0.5, fc='b', ec='None')
plt.show()

#for i in range(int(np.floor(duration * hz))):
	#LS.propogate(1) # command an ascent rate of 1 unit/sec
	#LS.propogate(0)

#print(LS.loon_history)

#LS.plot()
