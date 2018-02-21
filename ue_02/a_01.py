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
    f = lambda x, t: (x[1], -2*cos(x[2])/sin(x[2])*x[1]*x[3], x[3], sin(x[2])*cos(x[2])*(x[1])**2-sin(x[2]))

    res = odeint(f, (0, 0.1, 1, 0), t)

    x = map(lambda phi, psi: cos(phi)*sin(psi), res[:, 0], res[:, 2])
    y = map(lambda phi, psi: sin(phi)*sin(psi), res[:, 0], res[:, 2])
    z = map(lambda psi: -cos(psi), res[:, 2])

    figure = plt.figure()
    ax = figure.gca(projection='3d')
    ax.plot(x, y, z)

    plt.show()
