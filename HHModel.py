# By Bas Laarakker

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class HHModel():

    def __init__(self, gbar_K=36, gbar_Na=120, gbar_L=0.3, V_K=-12, V_Na=115, V_L=10.613, C_m=1.0, dt=0.01):
        """ Set the constants for the model; the default values are taken from the original paper by Hodgkin and Huxley """
        self.gbar_K = gbar_K
        self.gbar_Na = gbar_Na
        self.gbar_L = gbar_L

        self.V_K = V_K
        self.V_Na = V_Na
        self.V_L = V_L

        self.C_m = C_m

        self.dt = dt

    def alpha_n(self, V_m):
        """ Opening rate of a potassium gate """
        if (V_m == 10):
            return self.alpha_n(V_m+0.001)
        return 0.01*(10.0 - V_m)/(np.exp((10.0 - V_m) / 10.0) - 1.0)

    def beta_n(self, V_m):
        """ Closing rate of a potassium gate """
        return 0.125*np.exp(-V_m / 80.0)

    def alpha_m(self, V_m):
        """ Opening rate of a sodium gate """
        if (V_m == 25):
            return self.alpha_m(V_m+0.001)
        return 0.1*(25.0 - V_m)/(np.exp((25.0 - V_m) / 10.0) - 1.0)

    def beta_m(self, V_m):
        """ Closing rate of a sodium gate """
        return 4.0*np.exp(-V_m / 18.0)

    def alpha_h(self, V_m):
        """ Opening rate of inactivation variable of a sodium gate """
        return 0.07*np.exp(-V_m / 20.0)

    def beta_h(self, V_m):
        """ Closing rate of inactivation variable of a sodium gate """
        return 1.0 / (np.exp((30.0 - V_m) / 10.0) + 1.0)

    def I_K(self, V_m, n):
        """ Flow of potassium ions """
        return self.gbar_K * (n**4) * (V_m - self.V_K)

    def I_Na(self, V_m, m, h):
        """ Flow of sodium ions """
        return self.gbar_Na * (m**3) * h * (V_m - self.V_Na)

    def I_L(self, V_m):
        """ Leak current """
        return self.gbar_L * (V_m - self.V_L)

    def simple_ext_I(self, t):
        """ A constant injected current of 10 uA starting at 10% of time and ending at 90% of time """
        if (t < 0.1*self.total_time):
            return 0
        elif (t < 0.9*self.total_time):
            return 10
        else:
            return 0

    def set_init_conditions(self, V_m):
        n = self.alpha_n(V_m) / (self.alpha_n(V_m) + self.beta_n(V_m))
        m = self.alpha_m(V_m) / (self.alpha_m(V_m) + self.beta_m(V_m))
        h = self.alpha_h(V_m) / (self.alpha_h(V_m) + self.beta_h(V_m))
        return n, m, h
        
    def derivatives(self, y0, t):
        """ The differential equations governing the membrane potential """
        V_m, n, m, h = y0

        dVdt = (self.I_in(t) - (self.I_K(V_m, n) + self.I_Na(V_m, m, h) + self.I_L(V_m))) / self.C_m
        dndt = self.alpha_n(V_m) * (1 - n) - self.beta_n(V_m) * n
        dmdt = self.alpha_m(V_m) * (1 - m) - self.beta_m(V_m) * m
        dhdt = self.alpha_h(V_m) * (1 - h) - self.beta_h(V_m) * h

        return dVdt, dndt, dmdt, dhdt

    def run(self, y0, t, I_in = None, plot=False):
        """ Returns a matrix of I_inj, V, n, m and h at each timestep
        param y0: array of starting values [V, n, m, h]
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
        
        if (len(y0) == 1):
            V_m = y0[0]
            n, m, h = self.set_init_conditions(V_m)
            y0 = [V_m, n, m, h]
        
        sol = odeint(self.derivatives, y0, self.t_points)

        if plot:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 8), sharex=False)

            ax1.plot(self.t_points, I_arr, color="blue", linewidth=1)
            ax1.set_ylabel("Current (\u03bcA)")
            ax1.set_xlabel("Time (ms)")
            # ax1.set_yticks([0, 5, 10, 15])

            ax2.plot(self.t_points, sol[:, 0], color="red", linewidth=1)
            ax2.set_ylabel("Voltage (mV)")
            ax2.set_xlabel("Time (ms)")

            ax3.plot(self.t_points, sol[:, 1], label="n", linewidth=1)
            ax3.plot(self.t_points, sol[:, 2], label="m", linewidth=1)
            ax3.plot(self.t_points, sol[:, 3], label="h", linewidth=1)
            ax3.set_xlabel("Time (ms)")
            ax3.legend()

            ax4.plot(self.t_points, (self.gbar_K * (sol[:, 1]**4)), label="$G_{K}$", linewidth=1)
            ax4.plot(self.t_points, (self.gbar_Na * (sol[:, 2]**3) * sol[:, 3]), label="$G_{Na}$", linewidth=1)
            ax4.set_xlabel("Time (ms)")
            ax4.legend()

            plt.subplots_adjust(hspace=0.5)

            plt.show()


        concat = np.c_[I_arr, sol]

        return concat
