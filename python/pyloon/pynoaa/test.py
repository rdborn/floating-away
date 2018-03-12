from databringer import fetch
import matplotlib.pyplot as plt
import numpy as np

hrs = 0
while(True):
    data = fetch(38,-120+180,38,-120+180,hrs)
    print(len(data))
    # for d in data:
        # print("lat: " + str(d[0]) + "\tlon: " + str(d[1]) + "\talt: " + str(d[2]) + "\tvx: " + str(d[3]) + "\tvy: " + str(d[4]))
    # plot_idx = (data[:,2] > 10000)
    # plot_idx &= (data[:,2] < 22000)
    # data = data[plot_idx]
    v = data[:,3:5]
    for i in range(len(v)):
        # v[i] = v[i] / np.linalg.norm(v[i])
        pass
    print(hrs)
    plt.scatter(v[:,0], v[:,1], c=data[:,2],s=50,cmap='coolwarm')
    plt.plot(data[:,3],data[:,4])
    plt.axis('equal')
    plt.show()
    hrs += 3
    if hrs > 54:
        break
