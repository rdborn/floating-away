import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class LoonPathPlanner:
    def __init__(self, *args, **kwargs):
        field = kwargs.get('field') # wind field object
        res = kwargs.get('res')     # grid resolution in m
        x = np.linspace(0, field.xdim, field.xdim / res)
        y = np.linspace(0, field.ydim, field.ydim / res)
        z = np.linspace(0, field.zdim, field.zdim / res)
        XYZ = np.meshgrid(x, y, z)

        fx = np.zeros(len(x)*len(y)*len(z))
        fy = np.zeros(len(x)*len(y)*len(z))
        fz = np.zeros(len(x)*len(y)*len(z))
        idx = 0
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    mag, angle = field.get_flow(XYZ[0][i][j][k], XYZ[1][i][j][k], XYZ[2][i][j][k])
                    fx[idx] = fd * cos(angle)
            		fy[idx] = fd * sin(angle)
            		fz[idx] = 0
                    


        kernel = C(1.0, (1e-3, 1e-3)) * RBF(10, (1e-3, 1e-3))
        self.GPx = GPR(kernel=kernel, n_restarts_optimizer=9)
