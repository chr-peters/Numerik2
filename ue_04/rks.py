#! /usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np

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

def rks(f, T, x0, abbc, eps=1e-6, verbose=False, rho=0.9, qmin = 0.2, qmax = 5.):
    s = np.shape(abbc)[0]-2
    A = abbc[:s, 1:]
    b = abbc[s, 1:]
    b2 = abbc[-1, 1:]
    c = abbc[:s, 0]
    p = abbc[-2, 0]
    p2 = abbc[-1, 0]

    x0 = np.array(x0)
    t_res = [T[0]]
    x_res = [x0]

    cur_t = T[0]

    cur_h = (T[1] - T[0])*eps

    k_fehlberg = np.empty(shape=(np.shape(x0)[0]))

    # used to count function evaluations
    func_evals = 0
    
    while cur_t < T[1]:
        k = np.empty(shape=(s, np.shape(x0)[0]), dtype=np.float)

        cur_x = x_res[-1]
        cur_x2 = x_res[-1]

        # calculate the ks
        i = 0
        while i < s:

            # fehlberg trick
            if cur_t != T[0] and i == 0 and p2 > 0:
                k[0] = k_fehlberg[:]
            else:
                # calculate the inner sum
                j = 0
                sum = np.zeros(shape=(np.shape(x0)))
                while j < i:
                    sum += A[i, j]*k[j, :]
                    j+=1

                # set the current k
                k[i] = f(x_res[-1]+cur_h*sum,cur_t + c[i]*cur_h)
                func_evals = func_evals + 1

            # update phi
            cur_x = cur_x + cur_h * b[i] * k[i, :]
            cur_x2 = cur_x2 + cur_h * b2[i] * k[i, :]

            i+=1

        # estimate l
        sum = 0
        i = 0
        d = np.shape(cur_x)[0]
        while i < d:
            sum = sum + ((cur_x[i] - cur_x2[i])/(1 + max(abs(x_res[-1][i]), cur_x[i])))**2
            i = i + 1
        l = sqrt(1./d * sum)

        if l != 0:
            h_opt = cur_h * rho * (eps/l)**(1./(p2+1))
            h_opt = min(qmax * cur_h, max(qmin * cur_h, h_opt))
        else:
            h_opt = cur_h

        if l > eps:
            cur_h = h_opt
            continue

        x_res += [cur_x]
        cur_t += cur_h
        cur_h = h_opt
        t_res += [cur_t]

        # fehlberg trick
        k_fehlberg = k[-1, :]

    if verbose:
        print('number of function evaluations: '+str(func_evals))
    return np.array(x_res), t_res
