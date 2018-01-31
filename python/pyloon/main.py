from loonsim import LoonSim
import numpy as np

xdim = 5
ydim = 5
zdim = 20
hz = 20
duration = 6

LS = LoonSim(xdim, ydim, zdim)
LS.field.set_planar_flow(int(np.ceil(zdim/2 + 2)),1,np.pi/2)
LS.field.set_planar_flow(int(np.ceil(zdim/2 + 3)),1,np.pi)
LS.set_sample_rate(hz)

for i in range(int(np.floor(duration * hz))):
	LS.propogate(1) # command an ascent rate of 1 unit/sec
	#LS.propogate(0)

print(LS.loon_history)

LS.plot()
