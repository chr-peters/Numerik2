#! /usr/bin/env python2

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint

from math import *

from scipy import array

from explmm import expLMM, ABcoeff

if __name__ == '__main__':

    for problem in ['a','b']:
        if problem == 'a':
            t = np.arange(0, 100, 0.01)

            a = 0.001
            g = 100
            T = [0, 100]
            f = lambda x, t: a*x*(g-x)

            x0 = np.array([10])
            
            x_odeint = odeint(f, x0, t)

            k = 2

            A, B = ABcoeff(k)

            for plotnum, h in enumerate([20, 10, 5, 2, 1, 0.5]):
                x_adams_adams, t_adams_adams = expLMM(f, T, x0, k, A, B, h, method='adams')
                
                #x_adams_rk, t_adams_rk = expLMM(f, T, x0, k, A, B, h, method='rk')

                plt.subplot(2, 3, plotnum+1)
                plt.plot(t, x_odeint[:, 0])
                plt.plot(t_adams_adams, x_adams_adams[:, 0])
                #plt.plot(t_adams_rk, x_adams_rk[:, 0])
                plt.title('k = '+str(k)+', h = '+str(h))
                plt.xlabel('t')
                plt.ylabel('Anzahl')
                plt.legend(['odeint', 'adams using adams', 'adams using rk'])

            plt.tight_layout()
            plt.show()

        elif problem == 'b':
            t = np.arange(0, 30, 0.01)

            T = [0, 30]
            m1 = 10
            m2 = 1
            l1 = 1
            l2 = 0.75
            g = 9.81

            def f(x, t):
                A = np.array([[(m1+m2)*l1**2, m2*l1*l2*cos(x[0]-x[2])],
                              [m2*l1*l2*cos(x[0]-x[2]), m2*l2**2]])
                b = np.array([-m2*l1*l2*sin(x[0]-x[2])*x[3]**2-(m1+m2)*g*l1*sin(x[0]),
                              m2*l1*l2*sin(x[0]-x[2])*x[1]**2-m2*g*l2*sin(x[2])])
                res = np.linalg.solve(A, b)
                return [x[1], res[0], x[3], res[1]]

            x0 = np.array([1, 0, 1.5, 0])
            
            x_odeint = odeint(f, x0, t)

            k = 3

            A, B = ABcoeff(k)

            for plotnum, h in enumerate([0.05, 0.01, 0.001]):
                x_adams_adams, t_adams_adams = expLMM(f, T, x0, k, A, B, h, method='adams')
                #x_adams_rk, t_adams_rk = expLMM(f, T, x0, k, A, B, h, method='rk')

                plt.subplot(3, 2, 2*plotnum+1)
                plt.plot(t, x_odeint[:, 0])
                plt.plot(t_adams_adams, x_adams_adams[:, 0])
                #plt.plot(t_adams_rk, x_adams_rk[:, 0])
                plt.title('k = '+str(k)+', h = '+str(h))
                plt.xlabel('t')
                plt.ylabel('phi1')
                plt.legend(['odeint', 'adams using adams', 'adams using rk'])

                plt.subplot(3, 2, 2*plotnum+2)
                plt.plot(t, x_odeint[:, 2])
                plt.plot(t_adams_adams, x_adams_adams[:, 2])
                #plt.plot(t_adams_rk, x_adams_rk[:, 0])
                plt.title('k = '+str(k)+', h = '+str(h))
                plt.xlabel('t')
                plt.ylabel('phi2')
                plt.legend(['odeint', 'adams using adams', 'adams using rk'])

            plt.tight_layout()
            plt.show()
