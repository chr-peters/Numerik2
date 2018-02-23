#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint

from math import *

"""
NDGL1

Butcher-Schemata

Aufruf:  rkkoeff("euler")

Implementiert sind:
    euler, collatz, rk4, k38, rk43, dp54, rkf23

"""

from scipy import array

def rkkoeff(name):
    def euler():
        #  Betrag des letzten Koeffizienten in der ersten Spalte ist
        # Konsistenzordnung
        # Vorzeichen: + Fehlberg-Trick, - kein Fehlberg-Trick
        return array([[0,0], \
                      [-1,1] ], dtype=float)
                
    def collatz():
        #  Betrag des letzten Koeffizienten in der ersten Spalte ist
        # Konsistenzordnung
        # Vorzeichen: + Fehlberg-Trick, - kein Fehlberg-Trick
        return array([[   0 ,  0  , 0], \
                      [ 1./2, 1./2, 0], \
                      [  -2 ,   0 , 1] ], dtype=float) 
    
        
    def rk4():
        #  Betrag des letzten Koeffizienten in der ersten Spalte ist
        # Konsistenzordnung
        # Vorzeichen: + Fehlberg-Trick, - kein Fehlberg-Trick
        return array([[   0 ,   0 ,   0 ,   0 ,   0 ], \
                      [ 1./2, 1./2,   0 ,   0 ,   0 ], \
                      [ 1./2,   0 , 1./2,   0 ,   0 ], \
                      [   1 ,   0 ,   0 ,   1 ,   0 ], \
                      [  -4 , 1./6, 1./3, 1./3, 1./6]], dtype=float)
    
      
    def k38():
        #  Betrag des letzten Koeffizienten in der ersten Spalte ist
        # Konsistenzordnung
        # Vorzeichen: + Fehlberg-Trick, - kein Fehlberg-Trick
        return array([[   0 ,    0 ,   0 ,   0 ,   0 ],\
                      [ 1./3,  1./3,   0 ,   0 ,   0 ],\
                      [ 2./3, -1./3,   1 ,   0 ,   0 ],\
                      [   1 ,    1 ,  -1 ,   1 ,   0 ],\
                      [  -4 ,  1./8, 3./8, 3./8, 1./8]], dtype=float)
    
        
        
    def rk43():
        # abbc = rk43()
        #  Koeffizienten RKF 4(3)
        #  klassischer RK4 auf 5 Stufen aufgebohrt und 3.Ordnung eingebettet
        #  Deuflhard./Bornemann S.208
        #  Betrag der letzten beiden Koeffizienten in der ersten Spalte sind
        # Konsistenzordnungen
        # Vorzeichen: + Fehlberg-Trick, - kein Fehlberg-Trick        
        return array([\
             [  0 ,     0 ,     0 ,     0 ,     0,    0 ],\
             [1./2,   1./2,     0 ,     0 ,     0,    0 ],\
             [1./2,     0 ,   1./2,     0 ,     0,    0 ],\
             [  1 ,     0 ,     0 ,     1 ,     0,    0 ],\
             [  1 ,   1./6,   1./3,   1./3,   1./6,   0 ],\
             [  4 ,   1./6,   1./3,   1./3,   1./6,   0 ],\
             [  3 ,   1./6,   1./3,   1./3,     0 , 1./6]], dtype=float)


    def dp54():
        # abbc = dp54()
        #  Koeffizienten Dormand-Prince 5(4)
        #  Hairer./Wanner, Band 1, S. 171 (bzw. Deuflhard./Bornemann S.209)
        #  Betrag der letzten beiden Koeffizienten in der ersten Spalte sind
        # Konsistenzordnungen
        # Vorzeichen: + Fehlberg-Trick, - kein Fehlberg-Trick        
        return array([\
             [  0   ,       0      ,        0     ,        0      ,       0    ,        0       ,     0      ,   0  ],\
             [1./5  ,     1./5     ,        0     ,        0      ,       0    ,        0       ,     0      ,   0  ],\
             [3./10 ,     3./40    ,      9./40   ,        0      ,       0    ,        0       ,     0      ,   0  ],\
             [4./5  ,    44./45    ,    -56./15   ,     32./9     ,       0    ,        0       ,     0      ,   0  ],\
             [8./9  , 19372./6561  , -25360./2187 ,  64448./6561  ,  -212./729 ,        0       ,     0      ,   0  ],\
             [  1   ,  9017./3168  ,   -355./33   ,  46732./5247  ,    49./176 ,  -5103./18656  ,     0      ,   0  ],\
             [  1   ,    35./384   ,        0     ,    500./1113  ,   125./192 ,  -2187./6784   ,  11./84    ,   0  ],\
             [  5   ,    35./384   ,        0     ,    500./1113  ,   125./192 ,  -2187./6784   ,  11./84    ,   0  ],\
             [  4   ,  5179./57600 ,        0     ,   7571./16695 ,   393./640 , -92097./339200 , 187./2100  , 1./40]], dtype=float)


    def rkf23():
        # abbc = rkf23()
        #  Koeffizienten RKF2(3)
        #  Betrag der letzten beiden Koeffizienten in der ersten Spalte sind
        # Konsistenzordnungen
        # Vorzeichen: + Fehlberg-Trick, - kein Fehlberg-Trick               
        return array([\
             [ 0    ,       0    ,       0    ,      0 ],\
             [ 1    ,       1    ,       0    ,      0 ],\
             [1./2  ,     1./4   ,     1./4   ,      0 ],\
             [ 2    ,      1./2  ,     1./2   ,      0 ],\
             [ 3    ,      1./6  ,     1./6   ,    2./3]], dtype=float)
    
    def bs54():
        # abbc = bs54()
        #  Koeffizienten Bogacki-Shampine 5(4)
        #  An efficient Runge-Kutta (4,5) pair, Computers Math. Applic. VOl. 32, No. 6, pp 15-28
        #  Betrag der letzten beiden Koeffizienten in der ersten Spalte sind
        # Konsistenzordnungen
        # Vorzeichen: + Fehlberg-Trick, - kein Fehlberg-Trick               
        return array([\
             [ 0   ,        0   ,        0   ,        0   ,        0   ,        0   ,        0   ,        0  ,   0],\
             [1./6  ,      1./6   ,       0   ,        0   ,        0   ,        0   ,        0   ,        0 ,   0 ],\
             [6./27 ,     2./27   ,     4./27   ,      0   ,        0   ,        0   ,        0   ,        0 ,   0 ],\
             [3./7  ,     183./1372   ,-162./343 , 1053./1372   ,       0   ,        0   ,        0   ,      0 ,   0],\
             [2./3  ,   68./297 , -4./11 , 42./143 ,  1960./3861   ,     0   ,        0   ,        0  ],\
             
             [ 1   ,    9017./3168 ,  -355./33   , 46732./5247 ,    49./176  , -5103./18656   ,   0   ,        0  ],\
             [ 1   ,      35./384   ,     0   ,     500./1113 ,   125./192  , -2187./6784   ,  11./84   ,      0  ],\
             [ 5   ,      35./384   ,     0   ,     500./1113 ,   125./192  , -2187./6784   ,  11./84   ,      0  ],\
             [ 4   ,    5179./57600   ,   0   ,    7571./16695,   393./640  ,-92097./339200 , 187./2100   ,   1./40]], dtype=float)

    return eval(name+'()')
#    return eval(name)

def rkskalar(f, x0, T, abc, h):
    return np.array([[0, 0]]), [0]

if __name__ == '__main__':

    t = np.arange(0, 100, 0.01)

    a = 0.001
    g = 100
    T = [0, 100]
    f = lambda x, t: a*x*(g-x)

    x0 = 10

    x_odeint = odeint(f, x0, t)

    for plotnum, h in enumerate([20, 10, 5, 2, 1, 0.5]):

        x_euler, t_euler = rkskalar(f, x0, T, rkkoeff('euler'), h)
        x_collatz, t_collatz = rkskalar(f, x0, T, rkkoeff('collatz'), h)
        x_rk4, t_rk4 = rkskalar(f, x0, T, rkkoeff('rk4'), h)

        plt.subplot(2, 3, plotnum+1)
        plt.plot(t, x_odeint[:, 0])
        plt.plot(t_euler, x_euler[:, 0])
        plt.plot(t_collatz, x_collatz[:, 0])
        plt.plot(t_rk4, x_rk4[:, 0])
        plt.title('h = '+str(h))
        plt.xlabel('t')
        plt.ylabel('Anzahl')
        plt.legend(['odeint', 'euler', 'collatz', 'rk4'])

    plt.tight_layout()
    plt.show()
