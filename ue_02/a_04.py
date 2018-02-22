#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint

from math import *
import cmath

from a_03 import euler

if __name__ == '__main__':
    t = np.arange(0, 20, 0.01)

    k = 1
    
    u = 0
    v = 1

    for i, b in enumerate([3, 1, 0.1]):
        w = cmath.sqrt(b**2 - 4*k)
        x_exakt = map(lambda t: (u*(b+w)+2*v)/(2*w)*cmath.exp((-b+w)/2*t)-(u*(b-w)+2*v)/(2*w)*cmath.exp((-b-w)/2*t), t)

        f = lambda x, t: np.r_[x[1], -b*x[1]-k*x[0]]

        x_odeint = odeint(f, (u, v), t)

        plt.subplot(3, 1, i+1)
        plt.plot(t, x_exakt)
        plt.plot(t, x_odeint[:, 0])
        for h in [0.5, 0.1, 0.01]:
            x_euler, t_euler = euler(f, (u, v), [0, 20], h)
            plt.plot(t_euler, x_euler)

        plt.legend(['exakt', 'odeint', 'euler h=0.5', 'euler h=0.1', 'euler h=0.01'])
        plt.title('b = '+str(b))
        
    plt.show()
