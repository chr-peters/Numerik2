#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from math import *

from a_04 import rk, rkkoeff

if __name__ == '__main__':
    t = np.arange(0, 9, 0.01)

    T = [0, 9]

    mr = 1
    ms = 1e9
    mp = 1e8

    gamma = 1

    f = lambda x, t: [x[2], x[3],
                               -gamma*ms/np.linalg.norm(np.array([x[0], x[1]])-np.array([x[8], x[9]]))**3*(x[0]-x[8])-gamma*mp/np.linalg.norm(np.array([x[0], x[1]])-np.array([x[4], x[5]]))**3*(x[0]-x[4]),
                               -gamma*ms/np.linalg.norm(np.array([x[0], x[1]])-np.array([x[8], x[9]]))**3*(x[1]-x[9])-gamma*mp/np.linalg.norm(np.array([x[0], x[1]])-np.array([x[4], x[5]]))**3*(x[1]-x[5]),
                               x[6], x[7],
                               -gamma*ms/np.linalg.norm(np.array([x[4], x[5]])-np.array([x[8], x[9]]))**3*(x[4]-x[8])-gamma*mr/np.linalg.norm(np.array([x[4], x[5]])-np.array([x[0], x[1]]))**3*(x[4]-x[0]),
                               -gamma*ms/np.linalg.norm(np.array([x[4], x[5]])-np.array([x[8], x[9]]))**3*(x[5]-x[9])-gamma*mr/np.linalg.norm(np.array([x[4], x[5]])-np.array([x[0], x[1]]))**3*(x[5]-x[1]),
                               x[10], x[11],
                               -gamma*mr/np.linalg.norm(np.array([x[8], x[9]])-np.array([x[0], x[1]]))**3*(x[8]-x[0])-gamma*mp/np.linalg.norm(np.array([x[8], x[9]])-np.array([x[4], x[5]]))**3*(x[8]-x[4]),
                               -gamma*mr/np.linalg.norm(np.array([x[8], x[9]])-np.array([x[0], x[1]]))**3*(x[9]-x[1])-gamma*mp/np.linalg.norm(np.array([x[8], x[9]])-np.array([x[4], x[5]]))**3*(x[9]-x[5])]

    x0 = [1100, 0, -100, 0, 1000, 0, 0, 1000, -100, 0, 0, -100]

    x_odeint = odeint(f, x0, t)

    plt.subplot(3, 1, 1)
    plt.plot(x_odeint[:, 0], x_odeint[:, 1])
    plt.plot(x_odeint[:, 4], x_odeint[:, 5])
    plt.plot(x_odeint[:, 8], x_odeint[:, 9])
    plt.legend(['r', 'p', 's'])
    plt.title('odeint')

    x_rk4, t_rk4 = rk(f, x0, T, rkkoeff('rk4'), 0.1)
    plt.subplot(3, 1, 2)
    plt.plot(x_rk4[:, 0], x_rk4[:, 1])
    plt.plot(x_rk4[:, 4], x_rk4[:, 5])
    plt.plot(x_rk4[:, 8], x_rk4[:, 9])
    plt.legend(['r', 'p', 's'])
    plt.title('rk4, h = 0.1')

    x_rk4, t_rk4 = rk(f, x0, T, rkkoeff('rk4'), 0.01)
    plt.subplot(3, 1, 3)
    plt.plot(x_rk4[:, 0], x_rk4[:, 1])
    plt.plot(x_rk4[:, 4], x_rk4[:, 5])
    plt.plot(x_rk4[:, 8], x_rk4[:, 9])
    plt.legend(['r', 'p', 's'])
    plt.title('rk4, h = 0.01')

    plt.tight_layout()
    plt.show()
    
