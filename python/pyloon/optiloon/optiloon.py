import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C

class LoonPathPlanner:
    def __init__(self, *args, **kwargs):
        field = kwargs.get('field') # wind field object
        res = kwargs.get('res')     # grid resolution in m
        self.retrain(field=field, res=res)

    def retrain(self, *args, **kwargs):
        field = kwargs.get('field') # wind field object
        res = kwargs.get('res')     # grid resolution in m

        # Get training points
        x = np.linspace(0, field.xdim-1, int(np.floor(field.xdim / res)))
        y = np.linspace(0, field.ydim-1, int(np.floor(field.ydim / res)))
        z = np.linspace(0, field.zdim-1, int(np.floor(field.zdim / res)))
        xyz = np.vstack(map(np.ravel, np.meshgrid(x, y, z))).T

        # Get training observations
        fx = np.zeros(len(xyz))
        fy = np.zeros(len(xyz))
        fz = np.zeros(len(xyz))
        for i in range(len(xyz)):
            mag, angle = field.get_flow(xyz[i,0], xyz[i,1], xyz[i,2])
            fx[i] = mag * np.cos(angle)
            fy[i] = mag * np.sin(angle)
            fz[i] = 0

        # Set up our Gaussian Process (yay!)
        L = 1
        lo = 1e-3
        hi = 1e2
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[L, L, L], length_scale_bounds=(lo, hi), nu=1.5)
        self.GPx = GPR(kernel=kernel, n_restarts_optimizer=3)
        self.GPy = GPR(kernel=kernel, n_restarts_optimizer=3)
        self.GPz = GPR(kernel=kernel, n_restarts_optimizer=3)

        # Train our Gaussian Gaussian Process
        print("Training GPx")
        self.GPx.fit(xyz, fx.ravel())
        print("Training GPy")
        self.GPy.fit(xyz, fy.ravel())
        print("Training GPz")
        self.GPz.fit(xyz, fz.ravel())

        return True

    def predict(self, p):
        mean_fx, std_fx = self.GPx.predict(np.atleast_2d(p), return_std=True)
        mean_fy, std_fy = self.GPy.predict(np.atleast_2d(p), return_std=True)
        mean_fz, std_fz = self.GPz.predict(np.atleast_2d(p), return_std=True)
        return mean_fx, mean_fy, mean_fz, std_fx, std_fy, std_fz
