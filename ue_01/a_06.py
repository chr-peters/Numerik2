#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from math import *

if __name__ == '__main__':
    t = np.arange(0, 20, 0.01)
    k = 1
    for c in 'ab':

        if c=='a':
            f = lambda x, t: (x[1], -k*sin(x[0]))
        else:
            f = lambda x, t: (x[1], -k*x[0])

        phi = odeint(f, (0, 1.5), t)

        plt.subplot(121)
        plt.plot(t, phi[:,0])
        plt.plot(t, phi[:,1])
        plt.xlabel('t')
        plt.title('phi=0, phi\'=1.5')
        plt.legend(['Winkel', 'Winkelgeschwindigkeit'])

        plt.subplot(122)
        plt.plot(phi[:, 0], phi[:, 1])
        plt.title('phi=0, phi\'=1.5 Phasenplot')
        plt.xlabel('Winkel')
        plt.ylabel('Winkelgeschwindigkeit')

        plt.tight_layout()
        plt.show()
