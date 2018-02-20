#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

if __name__ == '__main__':
    alpha = lambda x: 2-x
    beta = 1
    gamma = lambda x: 1.1
    delta = 1
    f = lambda x, t: (alpha(x[0])*x[0] - beta * x[0] * x[1], -gamma(x[1])*x[1] + delta * x[0] * x[1])
    
    t = np.arange(0, 100, 0.01)

    x = odeint(f, (1, 1), t)
    plt.subplot(321)
    plt.plot(t, x[:, 0])
    plt.plot(t, x[:, 1])
    plt.xlabel('t')
    plt.ylabel('Anzahl')
    plt.legend(['beute', 'raeuber'])
    plt.title('beute=1, raeuber=1')

    plt.subplot(322)
    plt.plot(x[:, 0], x[:, 1])
    plt.xlabel('Beute')
    plt.ylabel('Raeuber')
    plt.title('beute=1, raeuber=1 Phasenplot')

    x = odeint(f, (2, 1), t)
    plt.subplot(323)
    plt.plot(t, x[:, 0])
    plt.plot(t, x[:, 1])
    plt.xlabel('t')
    plt.ylabel('Anzahl')
    plt.legend(['beute', 'raeuber'])
    plt.title('beute=2, raeuber=1')

    plt.subplot(324)
    plt.plot(x[:, 0], x[:, 1])
    plt.xlabel('Beute')
    plt.ylabel('Raeuber')
    plt.title('beute=2, raeuber=1 Phasenplot')

    x = odeint(f, (5, 1), t)
    plt.subplot(325)
    plt.plot(t, x[:, 0])
    plt.plot(t, x[:, 1])
    plt.xlabel('t')
    plt.ylabel('Anzahl')
    plt.legend(['beute', 'raeuber'])
    plt.title('beute=5, raeuber=1')

    plt.subplot(326)
    plt.plot(x[:, 0], x[:, 1])
    plt.xlabel('Beute')
    plt.ylabel('Raeuber')
    plt.title('beute=5, raeuber=1 Phasenplot')

    plt.tight_layout()
    
    plt.show()
