#! /usr/bin/env python2

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:27:55 2017

@author: Grajewski
"""

import matplotlib
matplotlib.use('Qt4Agg')

from pylab import *
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint

def RichtungsFeld(f, T, X):
    # Eingaben
    #f: rechte Seite der Differentialgleichung
    # T = [t0, te] : Zeitintervall
    # X = [x0, x1] : Wertebereich
    
    m = 20
    n = 20
    
    t0 = T[0]
    te = T[1]
    
    x0 = X[0]
    x1 = X[1]
    
    tvec = np.linspace(t0, te, m)
    xvec = np.linspace(x0, x1, n)
    
    tt, xx = np.meshgrid(tvec, xvec)

    T_component = np.ones([m,n])

    vecf = np.vectorize(f)

    X_component  = vecf(xx, tt)
    
    plt.figure()
    plt.quiver(tt, xx, T_component, X_component, color='0.5', angles='xy')

def f(x, t):
    a = 0.015
    g = 10
    return a*x*(g-x)

if __name__ == '__main__':
    RichtungsFeld(f, [0, 50], [0, 15])
    t = np.arange(0, 50, 0.01)
    res = odeint(f, [0, 1, 3, 15], t)
    plt.plot(t, res)
    plt.show()
