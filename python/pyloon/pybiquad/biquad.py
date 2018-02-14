import numpy as np

import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0],'..'))

from pyutils.pyutils import parsekw, hash3d, hash4d, rng

class Biquad:
    # __init__()
    # Master init function (gets called with every object instantiation).
    # Type of initialization is implied by the kwargs. If conflicting
    # initializations are detected, the default initialization is used.
    def __init__(self, *args, **kwargs):
        # Set flags indicating which kwargs were provided
        num = parsekw(kwargs, 'num', None)
        den = parsekw(kwargs, 'den', None)
        ic = parsekw(kwargs, 'ic', None)
        Fs = parsekw(kwargs, 'Fs', None)
        p = parsekw(kwargs, 'p', None)
        i = parsekw(kwargs, 'i', None)
        d = parsekw(kwargs, 'd', None)
        Fc = parsekw(kwargs, 'Fc', None)
        Q = parsekw(kwargs, 'Q', None)
        initialization = parsekw(kwargs, 'initialization', 'default')

        self.y = 0           # output
        self.n = 3           # length of coefficient vectors

        # Initialize based on the implied initialization method
        if initialization == 'standard':
            self.__init_standard__(num=num, den=den, ic=ic, Fs=Fs)
        elif initialization == 'pid':
            self.__init_pid__(p=p, i=i, d=d, Fs=Fs)
        elif initialization == 'bilinear':
            self.__init_bilinear__(Fs=Fs, Fc=Fc, Q=Q)
        else:
            self.__init_default__()

        self.Ts = 1.0 / self.Fs

        # Normalize coefficients
        self.__normalize__()

    # __str__()
    # Prints out the transfer function when object is passed as
    # argument to print()
    # @return transfer function represented as a string
    def __str__(self):
        return_str = "\n"
        return_str += "\t " + str(self.b[0]) + " + " + str(self.b[1]) + "z^-1 + " + str(self.b[2]) + "z^-2\n"
        return_str += "\t------------------------------------\n"
        return_str += "\t " + str(self.a[0]) + " + " + str(self.a[1]) + "z^-1 + " + str(self.a[2]) + "z^-2\n"
        return_str += "\n"
        return return_str

    # __normalize__()
    # Normalizes all coefficients by the leading denominator coefficient
    # (standard practice)
    # @return success/failure
    def __normalize__(self):
        if self.a[0] == 0:
            print("WARNING in __normalize__(): division by zero, no action taken.\n")
            return False
        for i in range(self.n):
            self.a[i] = self.a[i] / self.a[0]
            self.b[i] = self.b[i] / self.a[0]
        return True

    # __init_default__()
    # Initializes biquad filter with default coefficients and initial
    # conditions (i.e. zeros all around) (i.e. the trivial filter)
    def __init_default__(self):
        self.a = np.zeros(self.n)
        self.b = np.zeros(self.n)
        self.w = np.zeros(self.n)
        self.Fs = 1.0

    # __init_standard__()
    # Initializes biquad filter from vectors of coefficients, initial
    # conditions, and a sampling rate.
    # @param num np.array containing numerator coefficients (default: [0, 0, 0])
    # @param den np.array containing denominator coefficients (default: [0, 0, 0])
    # @param ic np.array containing initial conditions for delay registers (default: [0, 0, 0])
    # @param Fs sampling rate in Hz (default: 1 Hz)
    def __init_standard__(self, *args, **kwargs):
		# Set flags indicating which kwargs were provided
        num = kwargs.get('num', np.zeros(self.n))
        den = parsekw(kwargs, 'den', np.zeros(self.n))
        ic = parsekw(kwargs, 'ic', np.zeros(self.n))
        Fs = parsekw(kwargs, 'Fs', 1.0)

        self.a = den
        self.b = num
        self.w = ic
        self.Fs = Fs

    # __init_pid__()
    # from: https://portal.ku.edu.tr/~cbasdogan/Courses/Robotics/projects/Discrete_PID.pdf
    # Initializes biquad filter from a set of PID gains
    # @param p proportional gain (default: 0)
    # @param i integral gains (default: 0)
    # @param d derivative gains (default: 0)
    # @param Fs sampling rate (default: 1 Hz)
    def __init_pid__(self, *args, **kwargs):
		# Set flags indicating which kwargs were provided
        self.__init_default__()
        kp = parsekw(kwargs, 'p', 0.0)
        ki = parsekw(kwargs, 'i', 0.0)
        kd = parsekw(kwargs, 'd', 0.0)
        Fs = parsekw(kwargs, 'Fs', 1.0)
        self.Fs = Fs
        Ts = 1.0 / self.Fs
        
        # Generate coefficients from gains and sampling rate
        self.b[0] = kp + ki * Ts / 2 + kd / Ts
        self.b[1] = -kp + ki * Ts / 2 - 2 * kd / Ts
        self.b[2] = kd / Ts
        self.a[0] = 1
        self.a[1] = -1
        self.a[2] = 0

    # from: https://www.earlevel.com/main/2003/03/02/the-bilinear-z-transform/
    # Fc = corner frequency in Hz
    # Fs = sampling rate in Hz
    def __init_bilinear__(self, *args, **kwargs):
		# Set flags indicating which kwargs were provided
        Fc = parsekw(kwargs, 'Fc', 1.0)
        Q = parsekw(kwargs, 'Q', 1.0)
        Fs = parsekw(kwargs, 'Fs', 1.0)

        # Generate coefficients from sampling rate, corner frequency, and Q
        K = np.tan(np.pi * Fc / Fs)
        self.b[0] = K**2
        self.b[1] = 2 * self.b[0]
        self.b[2] = self.b[0]
        self.a[0] = K**2 + K / Q + 1
        self.a[1] = 2 * (K**2 - 1)
        self.a[2] = K**2 - K / Q + 1

    # update()
    # from: https://en.wikipedia.org/wiki/Digital_biquad_filter
    # Given an input x, this function updates the value at the output
    # of the biquad filter.
    # @param x input to biquad filter
    # @return updated output of biquad filter
    def update(self, x):
        self.w[0] = self.w[1]
        self.w[1] = self.w[2]
        self.w[2] = x - self.a[1] * self.w[1] - self.a[2] * self.w[0]
        self.y = self.b[0] * self.w[2] + self.b[1] * self.w[1] + self.b[2] * self.w[0]
        return self.y

    # get_curr_val()
    # Returns current value at the biquad filter's output
    # @return output of biquad filter
    def get_curr_val(self):
        return self.y

class Integrator(Biquad):
    def __init__(self, *args, **kwargs):
		# Set flags indicating which kwargs were provided
        ki = parsekw(kwargs, 'i', 1.0)
        Fs = parsekw(kwargs, 'Fs', 1.0)
        Biquad.__init__(self, i=ki, Fs=Fs, initialization='pid')

class DoubleIntegrator:
    def __init__(self, *args, **kwargs):
		# Set flags indicating which kwargs were provided
        ki = parsekw(kwargs, 'i', 1.0)
        Fs = parsekw(kwargs, 'Fs', 1.0)

        self.H1 = Integratot(i=ki, Fs=Fs)
        self.H2 = Integrator(i=1.0, Fs=Fs)

    def update(self, x):
        return self.H2.update(self.H1.update(x))

    def get_curr_val(self):
        return self.H2.get_curr_val()
