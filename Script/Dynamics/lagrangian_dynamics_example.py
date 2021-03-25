"""
## =========================================================================== ## 
MIT License
Copyright (c) 2020 Roman Parak
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## =========================================================================== ## 
Author   : Roman Parak
Email    : Roman.Parak@outlook.com
Github   : https://github.com/rparak
File Name: lagrangian_dynamics_example.py
## =========================================================================== ## 
"""

# System (Default Lib.)
import sys
# Numpy (Array computing Lib.) [pip3 install numpy]
import numpy as np
# Mtaplotlib (Visualization Lib.) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Integrate a system of ordinary differential equations (ODE) [pip3 install scipy]
from scipy.integrate import odeint

class Dynamics_Ctrl(object):
    def __init__(self, L, m, time, dt):
        # << PUBLIC >> #
        # Arm Length [m]
        self.L  = [L[0], L[1]] 
        # Arm Length (1/2) - Center of Gravity [m]
        self.lg = [L[0]/2, L[1]/2]
        # Mass [kg]
        self.m  = [m[0], m[1]]
        # Moment of Invertia [kg.m^2]
        self.I  = [(1/3)*(m[0])*(L[0]**2), (1/3)*(m[1])*(L[1]**2)]
        # Gravitational acceleration [m/s^2]
        self.g  = 9.81
        # Initial Time Parameters (Calculation)
        self.t = np.arange(0.0, time, dt)
        # << PRIVATE >> #
        # Axes and Label initialization.
        self.__ax = [0, 0, 0, 0]
        self.__y_label = [r'$\dot\theta_1$', r'$\ddot\theta_1$', r'$\dot\theta_2$', r'$\ddot\theta_2$']
        # Display (Plot) variables.
        self.__plt = plt
        self.__fig, ((self.__ax[0], self.__ax[1]), (self.__ax[2], self.__ax[3])) = self.__plt.subplots(2, 2)

    def __lagrangian_dynamics(self, input_p, time):
        """
        Description:
            For many applications with fixed-based robots we need to find a multi-body dynamics formulated as:

            M(\theta)\ddot\theta + b(\theta, \dot\theta) + g(\theta) = \tau

            M(\theta)                      -> Generalized mass matrix (orthogonal).
            \theta,\dot\theta,\ddot\theta  -> Generalized position, velocity and acceleration vectors.
            b(\theta, \dot\theta)          -> Coriolis and centrifugal terms.
            g(\theta)                      -> Gravitational terms.
            \tau                           -> External generalized forces.

            Euler-Lagrange equation:

            L = T - U

            T -> Kinetic Energy (Translation + Rotation Part): (1/2) * m * v^2 + (1/2) * I * \omega^2 -> with moment of invertia 
            U -> Potential Energy: m * g * h

        Args:
            (1) input_p [Float Array]: Initial position of the Robot (2 Joints) -> Theta_{1, 2} and 1_Derivation Theta_{1,2}
            (2) time [Float]: Derivation of the Time.
        Returns:
            (1): param 1, param 3 [Float]: 1_Derivation Theta_{1,2}
            (2): param 2, param 4 [Float]: 2_Derivation Theta_{1,2}
        """

        theta_1  = input_p[0]; theta_2  = input_p[2]
        dtheta_1 = input_p[1]; dtheta_2 = input_p[3]

        # Generalized mass matrix -> M(\theta)
        M_Mat   = np.matrix([
            [self.I[0] + self.I[1] + self.m[0] * (self.lg[0]**2) + self.m[1] * ((self.L[0]**2) + (self.lg[1]**2) + 2 * self.L[0] * self.lg[1] * np.cos(theta_2)), self.I[1] + self.m[1] * ((self.lg[1]**2) + self.L[0] * self.lg[1] * np.cos(theta_2))], 
            [self.I[1] + self.m[1] * ((self.lg[1]**2) + self.L[0] * self.lg[1] * np.cos(theta_2)), self.I[1] + self.m[1] * (self.lg[1]**2)]
        ])

        # Coriolis and centrifugal terms -> b(\theta, \dot\theta)
        b_Mat   = np.matrix([
            [(-1) * self.m[1] * self.L[0] * self.lg[1] * dtheta_2 * (2 * dtheta_1 + dtheta_2) * np.sin(theta_2)], 
            [self.m[1] * self.L[0] * self.lg[1] * (dtheta_1**2) *np.sin(theta_2)]
        ])

        # Gravitational terms -> g(\theta) 
        g_Mat   = np.matrix([
            [self.m[0] * self.g * self.lg[0] * np.cos(theta_1) + self.m[1] * self.g * (self.L[0] * np.cos(theta_1) + self.lg[1] * np.cos(theta_1 + theta_2))], 
            [self.m[1] * self.g * self.lg[1] * np.cos(theta_1 + theta_2)]
        ])

        # \tau -> External generalized forces.
        tau_Mat = np.matrix([[0.0], [0.0]])

        # Ordinary Differential Equations (ODE) -> From Motion Equation
        # {\ddotTheta_1, \ddotTheta_2}
        ode_r = np.linalg.inv(M_Mat).dot(-b_Mat - g_Mat) + tau_Mat

        return [dtheta_1, ode_r[0][0], dtheta_2, ode_r[1][0]]

    def display_result(self, input_p):
        """
        Description:
            Function for calculating and displaying the results of Lagrangian Dynamics Calculation.
        Args:
            (1) input_p [Float Array]: Initial position of the Robot (2 Joints) -> Theta_{1, 2} and 1_Derivation Theta_{1,2}
        """

        calc_r = odeint(self.__lagrangian_dynamics, input_p, self.t)

        self.__fig.suptitle('Lagrangian Dynamics: Example', fontsize = 50, fontweight ='normal')

        for i in range(len(self.__ax)):
            self.__ax[i].plot(self.t, calc_r[:, i])
            self.__ax[i].grid()
            self.__ax[i].set_xlabel(r'time [s]', fontsize=20)
            self.__ax[i].set_ylabel(self.__y_label[i], fontsize=20)

        # Set additional parameters for successful display of the robot environment
        self.__plt.show()

def main():
    # Initialization of the Class (Control Dynamics - Lagrangian)
    # Input:
    #   (1) Length of Arms (Link 1, Link2) [Float Array]
    #   (2) Mass
    #   (3) Time [INT]
    #   (4) Derivation of the Time [Float]
    # Example:
    #   x = Dynamics_Ctrl([1.0, 1.0], [1.25, 2.0], 10, 0.1)

    lD_c = Dynamics_Ctrl([0.3, 0.25], [1.0, 1.0], 10, 0.01)

    # Initial position of the Robot (2 Joints) -> Theta_{1, 2} and 1_Derivation Theta_{1,2}
    initial_p = [0.1*np.pi, 0.0, 0.1*np.pi, 0.0]

    # Display the result of the calculation:
    # The figure with the resulting 1_Derivation Theta_{1,2}, 2_Derivation Theta_{1,2}
    lD_c.display_result(initial_p)

if __name__ == '__main__':
    sys.exit(main())


