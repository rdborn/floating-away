from loonsim import LoonSim
import numpy as np

xdim = 50
ydim = 50
zdim = 50
hz = 20
duration = 6

LS = LoonSim(xdim=xdim, ydim=ydim, zdim=zdim, Fs=hz)
LS.field.set_planar_flow(int(np.ceil(zdim/2 + 2)),1,0)
LS.field.set_planar_flow(int(np.ceil(zdim/2 + 3)),np.sqrt(2),np.pi)
LS.field.set_planar_flow(int(np.ceil(zdim/2 + 4)),1,0)

for i in range(int(np.floor(duration * hz))):
	LS.propogate(1) # command an ascent rate of 1 unit/sec
	#LS.propogate(0)

print(LS.loon_history)

LS.plot()
