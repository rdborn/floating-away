from databringer import fetch
import matplotlib.pyplot as plt
import numpy as np

hrs = 0
while(True):
    data = fetch(36,-122+180,38,-120+180,hrs)
    print(len(data))
    # HACK OUT THE WEIRD ERRONEOUS ENTRIES
    for d in data:
        coord = d[0:3]
        index = 1 * (data[:,0:3] == coord).all(axis=1)
        if sum(index) > 1:
            for i, idx in enumerate(index):
                if idx == 1:
                    index[i] = 0
                    break
            index = (index == 0)
            data = data[index]
        if abs(d[2] - 16975.9752451) < 1e-3:
            index = (index == 0)
            data = data[index]
    # for d in data:
        # print("lat: " + str(d[0]) + "\tlon: " + str(d[1]) + "\talt: " + str(d[2]) + "\tvx: " + str(d[3]) + "\tvy: " + str(d[4]))
    plot_idx = (data[:,2] > 10000)
    plot_idx &= (data[:,2] < 22000)
    data = data[plot_idx]
    v = data[:,3:5]
    for i in range(len(v)):
        v[i] = v[i] / np.linalg.norm(v[i])
    print(hrs)
    plt.scatter(v[:,0], v[:,1], c=data[:,2],s=50,cmap='coolwarm')
    # plt.plot(data[:,3],data[:,4])
    plt.axis('equal')
    plt.show()
    hrs += 3
    if hrs > 54:
        break
