#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from math import *

if __name__ == '__main__':
    t = np.arange(0, 20, 0.01)
    k=1

    tmp, plotgrid = plt.subplots(3, 4)

    for line, b in enumerate([0.1, 1, 3]):
        f_linear = lambda x, t: (x[1], -b*x[1]-k*x[0])
        f_nonlinear = lambda x, t: (x[1], -b*x[1]-k*sin(x[0]))

        for col, problem in enumerate(['linear', 'non-linear']):
            if problem == 'linear':
                f = f_linear
            else:
                f = f_nonlinear

            phi = odeint(f, (0, 1.5), t)

            cur_plot = plotgrid[line, col*2]
            cur_plot.set_title('b = '+str(b)+', '+problem)
            cur_plot.plot(t, phi[:,0])
            cur_plot.plot(t, phi[:,1])
            cur_plot.set_xlabel('t')
            cur_plot.legend(['Winkel', 'Winkelgeschwindigkeit'])

            cur_plot = plotgrid[line, col*2+1]
            cur_plot.set_title('b = '+str(b)+', '+problem+' Phasenplot')
            cur_plot.plot(phi[:, 0], phi[:,1])
            cur_plot.set_xlabel('Winkel')
            cur_plot.set_ylabel('Winkelgeschwindigkeit')

    plt.tight_layout()
    plt.show()
            
