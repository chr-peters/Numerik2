#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from a_04 import rk

from math import *

if __name__ == '__main__':
    t = np.arange(0, 20, 0.01)

    k = 1

    b = 0.1
    
    u = 1.5
    v = 0

    collatz_mat = np.array([[0, 0, 0], [0.5, 0.5, 0], [0, 0, 1]])
    rk_mat = np.array([[0, 0, 0, 0, 0],
                       [0.5, 0.5, 0, 0, 0],
                       [0.5, 0, 0.5, 0, 0],
                       [1, 0, 0, 1, 0],
                       [0, 1/6., 1/3., 1/3., 1/6.]])
    rk38_mat = np.array([[0, 0, 0, 0, 0],
                         [1/3., 1/3., 0, 0, 0],
                         [2/3., -1/3., 1, 0, 0],
                         [1, 1, -1, 1, 0],
                         [0, 1/8., 3/8., 3/8., 1/8.]])

    tmp, plotgrid = plt.subplots(3)

    f = lambda x, t: np.r_[x[1], -b*x[1]-k*x[0]]

    for row, h in enumerate([0.25, 0.1, 0.05]):

        x_odeint = odeint(f, [u, v], t)
        x_collatz, t_collatz = rk(f, [u, v], [0, 20], collatz_mat, h)
        x_rk, t_rk = rk(f, [u, v], [0, 20], rk_mat, h)
        x_rk38, t_rk38 = rk(f, [u, v], [0, 20], rk38_mat, h)

        cur_plot = plotgrid[row]
        cur_plot.plot(t, x_odeint[:, 0])
        cur_plot.plot(t_collatz, x_collatz[:,0])
        cur_plot.plot(t_rk, x_rk[:,0])
        cur_plot.plot(t_rk38, x_rk38[:,0])
        cur_plot.set_xlabel('t')
        cur_plot.set_ylabel('Winkel')
        cur_plot.set_title('b = '+str(b)+', h = '+str(h))
        cur_plot.legend(['odeint', 'collatz', 'rk', 'rk38'])

    plt.tight_layout()
    plt.show()
