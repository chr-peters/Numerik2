#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint

from math import *

if __name__ == '__main__':
    t = np.arange(0, 100, 0.01)

    sigma = 10
    b = 8./3
    r = 28

    f = lambda x, t: (-sigma*x[0] + sigma*x[1], -x[0]*x[2] + r*x[0] - x[1], x[0]*x[1] - b*x[2])

    x0 = (-8, 8, r-1)
    x = odeint(f, x0, t)

    figure = plt.figure()
    ax = figure.gca(projection='3d')
    ax.plot(x[:, 0], x[:, 1], x[:, 2])

    x0 = (-9, 9, r-2)
    x = odeint(f, x0, t)
    ax.plot(x[:, 0], x[:, 1], x[:, 2])

    x0 = (-7, 7, r-23)
    x = odeint(f, x0, t)
    ax.plot(x[:, 0], x[:, 1], x[:, 2])

    plt.show()

