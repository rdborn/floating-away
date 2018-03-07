import struct

file = "./data/gfs_4_20180225_0000_000/L1/C-100_100.gribp"
f = open(file,'rb')
N = 100
for i in range(N):
    data = f.read(20)
    (lat, lon, u, v, t) = struct.unpack('>fffff',data)
    print("lat: " + str(lat) + "\tlon: " + str(lon) + "\tu: " + str(u) + "\tv: " + str(v) + "\tt: " + str(t))
print(len(data))
