#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint

from math import *

def collatz(f, x0, T, h):
    t_res = [T[0]]
    x_res = [x0]

    cur_t = T[0]

    while cur_t < T[1]:
        x_res += [x_res[-1] + h*f(x_res[-1] + h/2 * f(x_res[-1], cur_t), cur_t + h/2.)]
        cur_t += h
        t_res += [cur_t]

    return np.array(x_res), t_res

def heun(f, x0, T, h):
    t_res = [T[0]]
    x_res = [x0]

    cur_t = T[0]

    while cur_t < T[1]:
        x_res += [x_res[-1] + h/2*(f(x_res[-1], cur_t) + f(x_res[-1] + h*f(x_res[-1], cur_t), cur_t+h))]
        cur_t += h
        t_res += [cur_t]

    return np.array(x_res), t_res

if __name__ == '__main__':
    t = np.arange(0, 20, 0.01)

    k = 1
    
    u = 1.5
    v = 0

    tmp, plotgrid = plt.subplots(3, 3)

    for line, b in enumerate([0.1, 1, 3]):

        f = lambda x, t: np.r_[x[1], -b*x[1]-k*x[0]]

        for col, h in enumerate([0.5, 0.1, 0.01]):

            x_odeint = odeint(f, [u, v], t)
            x_collatz, t_collatz = collatz(f, [u, v], [0, 20], h)
            x_heun, t_heun = heun(f, [u, v], [0, 20], h)

            # calc the mean squared error
            mean_squared = sum((map(lambda x1, x2: (x1-x2)**2, x_collatz[:,0], x_heun[:,0])))/len(t_collatz)
            print('Mean squared error between collatz and heun for b = '+str(b)+' and h = '+str(h)+' is: '+str(mean_squared))

            cur_plot = plotgrid[line, col]
            cur_plot.plot(t, x_odeint[:, 0])
            cur_plot.plot(t_collatz, x_collatz[:,0])
            cur_plot.plot(t_heun, x_heun[:,0])
            cur_plot.set_xlabel('t')
            cur_plot.set_ylabel('Winkel')
            cur_plot.set_title('b = '+str(b)+', h = '+str(h))
            cur_plot.legend(['odeint', 'collatz', 'heun'])

    plt.tight_layout()
    plt.show()
