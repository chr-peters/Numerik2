#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from math import *

from rks import rks, rkkoeff

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

    for task in ['a', 'b']:
        if task == 'a':
            x0 = [1100, 0, -100, 0, 1000, 0, 0, 1000, -100, 0, 0, -100]
        else:
            x0 = [1100, 200, -80, -20, 1000, 0, 0, 1000, -100, 0, 0, -100]

        x_odeint = odeint(f, x0, t)

        plt.suptitle(task)

        plt.subplot(2, 2, 1)
        plt.plot(x_odeint[:, 0], x_odeint[:, 1])
        plt.plot(x_odeint[:, 4], x_odeint[:, 5])
        plt.plot(x_odeint[:, 8], x_odeint[:, 9])
        plt.legend(['r', 'p', 's'])
        plt.title('odeint')

        print('rk23, eps = 1e-6')
        x_rk23, t_rk23 = rks(f, T, x0, rkkoeff('rkf23'), eps=1e-6, verbose = True)
        plt.subplot(2, 2, 2)
        plt.plot(x_rk23[:, 0], x_rk23[:, 1])
        plt.plot(x_rk23[:, 4], x_rk23[:, 5])
        plt.plot(x_rk23[:, 8], x_rk23[:, 9])
        plt.legend(['r', 'p', 's'])
        plt.title('rf23')

        print('rk43, eps = 1e-6')
        x_rk43, t_rk43 = rks(f, T, x0, rkkoeff('rk43'), eps=1e-6, verbose = True)
        plt.subplot(2, 2, 3)
        plt.plot(x_rk43[:, 0], x_rk43[:, 1])
        plt.plot(x_rk43[:, 4], x_rk43[:, 5])
        plt.plot(x_rk43[:, 8], x_rk43[:, 9])
        plt.legend(['r', 'p', 's'])
        plt.title('rk43')

        print('dp54, eps = 1e-6')
        x_dp54, t_dp54 = rks(f, T, x0, rkkoeff('dp54'), eps=1e-6, verbose = True)
        plt.subplot(2, 2, 4)
        plt.plot(x_dp54[:, 0], x_dp54[:, 1])
        plt.plot(x_dp54[:, 4], x_dp54[:, 5])
        plt.plot(x_dp54[:, 8], x_dp54[:, 9])
        plt.legend(['r', 'p', 's'])
        plt.title('dp54')

        plt.tight_layout()
        plt.show()
    
