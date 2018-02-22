#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint

from math import *

"""
Numerik 2, explizites Euler-Verfahren 


"""

from scipy import array, append, vstack

def euler(f, u0, T, h):
    """ 
    uv, tv = euler(f, u0, T, h)
    
    Euler-Verfahren mit fester Schrittweite
       
    IN:
        f(u,t) : muss array auf array abbilden
        
        T      : [t0,te]
        
        u0     : Startwert (array)
        
        h      : Zeitschrittweite
        
    OUT:
        uv     : Loesungskomponenten (Spalten) des arrays
        tv     : array
        
    t0 und te sind sicher im Output
    """
    
    # Start- und Endzeit ermitteln
    t0 = T[0]
    te = T[1]
    
    # Anfangswert ggf. auf Zeilenvektor transformieren
    u0 = array(u0).flatten()
    
    # Anfangszeit und -wert in Loesungsvektor eintragen
    tv = t0
    uv = u0.T
    
    
    # eigentliches Verfahren
    t = t0
    u = u0
    
    ttol = 1e-8 * h
    
    while t<te:
    # Letzten Schritt evtl. verkuerzen, damit man genau auf te landet
        tn = t + h
        if tn > te-ttol:
            tn = te
            h  = tn - t
    
    # Euler-Schritt    
        un = u + h * f(u,t)
    
    # neuer Zeitpunkt und neue Naeherung uebernehmen    
        t  = tn
        u  = un
        
    # aktuelles Ergebnis an den Ergbnisvektor anhaengen
        tv = append(tv, t)
        uv = vstack((uv,  u))
    
    return (uv, tv)

if __name__ == '__main__':
    t = np.arange(0, 100, 0.01)

    a = 0.001
    g = 100

    f = lambda x, t: a*x*(g-x)

    for x0 in [10, 200]:

        x_odeint = odeint(f, x0, t)
        x_exakt = map(lambda t: g / (1 + exp(-a*g*t)*(g-x0)/x0), t)

        plt.plot(t, x_odeint)
        plt.plot(t, x_exakt)
        plt.legend(['odeint', 'exakt'])

        for h in [10, 2, 1, 0.5]:
            x_euler, t_euler = euler(f, x0, (0, 100), h)
            plt.plot(t_euler, x_euler)

        plt.legend(['odeint', 'exakt', 'euler h=10', 'euler h=2', 'euler h=1', 'euler h=0.5'])
        plt.show()
