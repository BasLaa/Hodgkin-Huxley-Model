import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class Single_Ion_Model():

    def __init__(self, gbar_K=36, V_K=-12, C_m=1.0):
        """ Set the constants for the model; the default values are taken from the original paper """
        self.gbar_K = gbar_K
        self.V_K = V_K
        self.C_m = C_m

        self.dt = 0.01

    def alpha_n(self, V_m):
        """ Opening rate of a potassium gate """
        return 0.01*(10.0 - V_m)/(np.exp((10.0 - V_m) / 10.0) - 1.0)

    def beta_n(self, V_m):
        """ Closing rate of a potassium gate """
        return 0.125*np.exp(-V_m / 80.0)

    def I_K(self, V_m, n):
        """ Flow of potassium ions """
        return self.gbar_K * (n**4) * (V_m - self.V_K)

    def simple_ext_I(self, t):
        """ A constant injected current of 50 mA starting at 10% of time and ending at 90% of time """
        if (t < 0.1*self.total_time):
            return 0
        elif (t < 0.9*self.total_time):
            return 50
        else:
            return 0

    def derivatives(self, y0, t):
        """ The differential equations governing the membrane potential """
        V_m, n = y0

        dVdt = (self.I_in(t) - self.I_K(V_m, n)) / self.C_m
        dndt = self.alpha_n(V_m) * (1 - n) - self.beta_n(V_m) * n

        return dVdt, dndt

    def run(self, y0, t, I_in = None, plot=False):
        """ Returns a matrix of I_inj, V and n at each timestep
        param y0: array of starting values [V, n]
        param t: array of timepoints to integrate OR an integer/float representing length of simulation in milliseconds
        param I_in: a function of time for the injected current, if None uses constant current function simple_ext_I()
        """

        if (isinstance(t, (int, float))):
            self.total_time = t
            self.t_points = np.arange(0, t, self.dt)
        else:
            self.total_time = round(t[np.size(t)-1])
            self.t_points = t

        if (I_in == None):
            self.I_in = self.simple_ext_I
        else:
            self.I_in = I_in

        I_arr = np.array([self.I_in(x) for x in self.t_points])
        sol = odeint(self.derivatives, y0, self.t_points)

        if (plot):
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 7.5), sharex=True)
            ax1.plot(self.t_points, I_arr, color="blue")
            ax1.set_ylabel("Current (mA)")

            ax2.plot(self.t_points, sol[:, 0], color="red")
            ax2.set_ylabel("Voltage (mV)")

            ax3.plot(self.t_points, sol[:, 1], label="n", color="#332288")
            ax3.legend()

            ax4.plot(self.t_points, (self.gbar_K * (sol[:, 1]**4)), label="$G_{K}$")
            ax4.set_xlabel("Time (ms)")
            ax4.legend()

            plt.show()


        concat = np.c_[I_arr, sol]

        return concat

if __name__ == '__main__':
    model = Single_Ion_Model()

    # starting values
    y0 = [0, 0.05] # [V0, n0]
    t = np.arange(0, 100, 0.01) # time steps, can also just use an integer

    # input current function
    def I_in(t):
        return 20*np.sin(0.25*t)

    # This will run it with a standard constant input current
    # plot=True automatically creates some plots
    # res = model.run(y0, t, plot=True)

    # res is a matrix of the form [I_in, V, n, m, h]

    # This will use the defined current function
    res = model.run(y0, t, I_in, plot=True)
