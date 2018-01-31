import numpy as np

class Biquad:
    a = np.zeros(3) # denominator coefficients
    b = np.zeros(3) # numerator coefficients
    w = np.zeros(3) # delay registers
    y = 0           # output
    n = 3           # length of coefficient vectors
    Ts = 1          # sampling rate

    # __init__()
    # Master init function (gets called with every object instantiation).
    # Type of initialization is implied by the kwargs. If conflicting
    # initializations are detected, the default initialization is used.
    def __init__(self, *args, **kwargs):
        # Set flags indicating which kwargs were provided
        num = kwargs.get('num') != None
        den = kwargs.get('den') != None
        ic = kwargs.get('ic') != None
        Fs = kwargs.get('Fs') != None
        p = kwargs.get('p') != None
        i = kwargs.get('i') != None
        d = kwargs.get('d') != None
        Fc = kwargs.get('Fc') != None
        Q = kwargs.get('Q') != None

        # Determine initialization flavor
        standard = num or den or ic
        pid = p or i or d
        bilinear = Fs and (Fc or Q)
        default = False

        # Detect whether multiple flavors are requested
        neopolitan = (standard and pid) or (standard and bilinear) or (bilinear and pid)
        if neopolitan:
            print("WARNING: multiple initialization methods requested. Initializing with defaults.\n")
            default = True

        # Initialize based on the implied initialization method
        if default:
            self.__init_default__()
        elif standard:
            self.__init_standard__(num=kwargs.get('num'), den=kwargs.get('den'), ic=kwargs.get('ic'))
        elif pid:
            self.__init_pid__(p=kwargs.get('p'), i=kwargs.get('i'), d=kwargs.get('d'), Fs=kwargs.get('Fs'))
        elif bilinear:
            self.__init_bilinear__(Fs=kwargs.get('Fs'), Fc=kwargs.get('Fc'), Q=kwargs.get('Q'))

        # Set sampling rate
        if Fs:
            self.set_sampling_rate(kwargs.get('Fs'))
        else:
            self.set_sampling_rate(1.0)

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

    # __init_standard__()
    # Initializes biquad filter from vectors of coefficients, initial
    # conditions, and a sampling rate.
    # @param num np.array containing numerator coefficients (default: [0, 0, 0])
    # @param den np.array containing denominator coefficients (default: [0, 0, 0])
    # @param ic np.array containing initial conditions for delay registers (default: [0, 0, 0])
    # @param Fs sampling rate in Hz (default: 1 Hz)
    def __init_standard__(self, *args, **kwargs):
		# Set flags indicating which kwargs were provided
        num = kwargs.get('num') != None
        den = kwargs.get('den') != None
        ic = kwargs.get('ic') != None
        Fs = kwargs.get('Fs') != None

        # Set coefficients and initial conditions
        self.a = kwargs.get('num') if num else np.zeros(self.n)
        self.b = kwargs.get('den') if den else np.zeros(self.n)
        self.w = kwargs.get('ic') if ic else np.zeros(self.n)

    # __init_pid__()
    # from: https://portal.ku.edu.tr/~cbasdogan/Courses/Robotics/projects/Discrete_PID.pdf
    # Initializes biquad filter from a set of PID gains
    # @param p proportional gain (default: 0)
    # @param i integral gains (default: 0)
    # @param d derivative gains (default: 0)
    # @param Fs sampling rate (default: 1 Hz)
    def __init_pid__(self, *args, **kwargs):
		# Set flags indicating which kwargs were provided
        p = kwargs.get('p') != None
        i = kwargs.get('i') != None
        d = kwargs.get('d') != None
        Fs = kwargs.get('Fs') != None

        # Extract gains and sampling rate
        kp = kwargs.get('p') if p else 0.0
        ki = kwargs.get('i') if i else 0.0
        kd = kwargs.get('d') if d else 0.0
        Ts = 1.0 / kwargs.get('Fs') if Fs else 1.0

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
        Fc = kwargs.get('Fc') != None
        Q = kwargs.get('Q') != None
        Fs = kwargs.get('Fs') != None

        # Extract parameters
        omega_s = kwargs.get('Fs') if Fs else 1.0   # Sampling rate in Hz       [default: 1 Hz]
        omega_c = kwargs.get('Fc') if Fc else 1.0   # Corner frequency in Hz    [default: 1 Hz]
        _Q = kwargs.get('Q') if Q else 1.0          #

        # Generate coefficients from sampling rate, corner frequency, and Q
        K = np.tan(np.pi * omega_c / omega_s)
        self.b[0] = K**2
        self.b[1] = 2 * self.b[0]
        self.b[2] = self.b[0]
        self.a[0] = K**2 + K / _Q + 1
        self.a[1] = 2 * (K**2 - 1)
        self.a[2] = K**2 - K / _Q + 1

    # set_sampling_rate()
    # If the desired sampling rate is valid, the object's sampling
    # is updated accordingly.
    # @param Fs desired sampling rate in Hz
    # @return success/failure
    def set_sampling_rate(self, Fs):
        if Fs <= 0:
            print("WARNING in set_sampling_rate(): sampling rate must be positive. No action taken.\n")
            return False
        self.Ts = 1.0 / Fs
        return True

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

class Integrator:
    def __init__(self, *args, **kwargs):
		# Set flags indicating which kwargs were provided
        ki = kwargs.get('i') != None
        Fs = kwargs.get('Fs') != None

        k = kwargs.get('i') if ki else 1.0
        f = kwargs.get('Fs') if Fs else 1.0

        self.H = Biquad(i=k, Fs=f)

    def update(self, x):
        return self.H.update(x)

    def get_curr_val(self):
        return self.H.get_curr_val()

class DoubleIntegrator:
    def __init__(self, *args, **kwargs):
		# Set flags indicating which kwargs were provided
        ki = kwargs.get('i') != None
        Fs = kwargs.get('Fs') != None

        k = kwargs.get('i') if ki else 1.0
        f = kwargs.get('Fs') if Fs else 1.0

        self.H1 = Biquad(i=k, Fs=f)
        self.H2 = Biquad(i=1.0, Fs=f)

    def update(self, x):
        return self.H2.update(self.H1.update(x))

    def get_curr_val(self):
        return self.H2.get_curr_val()
