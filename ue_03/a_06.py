#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from a_04 import rk, rkkoeff

from math import *

if __name__ == '__main__':
    t = np.arange(0, 30, 0.01)

    T = [0, 30]

    g = 9.81
    l1 = 1
    l2 = 0.75
    m1 = 10
    m2 = 1

    def f(x, t):
        A = np.array([[(m1+m2)*l1**2, m2*l1*l2*cos(x[0]-x[2])],
                      [m2*l1*l2*cos(x[0]-x[2]), m2*l2**2]])
        b = np.array([-m2*l1*l2*sin(x[0]-x[2])*x[3]**2-(m1+m2)*g*l1*sin(x[0]),
                      m2*l1*l2*sin(x[0]-x[2])*x[1]**2-m2*g*l2*sin(x[2])])
        res = np.linalg.solve(A, b)
        return [x[1], res[0], x[3], res[1]]


    x0 = [1, 0, 1.5, 0]

    for plotnum, h in enumerate([0.05, 0.01, 0.005]):

        x_odeint = odeint(f, x0, t)
        x_rk4, t_rk4 = rk(f, x0, T, rkkoeff('rk4'), h)

        plt.subplot(3, 2, 2*plotnum+1)
        plt.plot(t, x_odeint[:, 0])
        plt.plot(t_rk4, x_rk4[:,0])
        plt.xlabel('t')
        plt.ylabel('phi1')
        plt.title('h = '+str(h)+', phi1')
        plt.legend(['odeint', 'rk4'])

        plt.subplot(3, 2, 2*plotnum+2)
        plt.plot(t, x_odeint[:, 2])
        plt.plot(t_rk4, x_rk4[:,2])
        plt.xlabel('t')
        plt.ylabel('phi2')
        plt.title('h = '+str(h)+', phi2')
        plt.legend(['odeint', 'rk4'])

    plt.tight_layout()
    plt.show()
