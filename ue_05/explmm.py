from scipy import array
import sys
import numpy as np

def ABcoeff(n):

    Alpha_AB=np.array([[-1.0, 1.0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  ],\
                          [ 0  ,-1.0, 1.0, 0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  ],\
                          [ 0  , 0  ,-1.0, 1.0, 0  , 0  , 0  , 0  , 0  , 0  , 0  ],\
                          [ 0  , 0  ,   0,-1.0, 1.0, 0  , 0  , 0  , 0  , 0  , 0  ],\
                          [ 0  , 0  ,   0, 0  ,-1.0, 1.0, 0  , 0  , 0  , 0  , 0  ],\
                          [ 0  , 0  ,   0, 0  , 0  ,-1.0, 1.0, 0  , 0  , 0  , 0  ],\
                          [ 0  , 0  ,   0, 0  , 0  , 0  ,-1.0, 1.0, 0  , 0  , 0  ],\
                          [ 0  , 0  ,   0, 0  , 0  , 0  , 0  ,-1.0, 1.0, 0  , 0  ],\
                          [ 0  , 0  ,   0, 0  , 0  , 0  , 0  , 0  ,-1.0, 1.0, 0  ],\
                          [ 0  , 0  ,   0, 0  , 0  , 0  , 0  , 0  , 0  ,-1.0, 1.0]], dtype=float)

    Beta_AB=np.array([[       1.        ,          0        ,         0       ,          0       ,           0       ,          0       ,          0       ,          0       ,           0       ,       0,        0],\
                         [     -1./2       ,        3./2       ,         0       ,          0       ,           0       ,          0       ,          0       ,          0       ,           0       ,       0,        0],\
                         [      5./12      ,       -4./3       ,      23./12     ,          0       ,           0       ,          0       ,          0       ,          0       ,           0       ,       0,        0],\
                         [     -3./8       ,       37./24      ,     -59./24     ,       55./24     ,           0       ,          0       ,          0       ,          0       ,           0       ,       0,        0],\
                         [    251./720     ,     -637./360     ,     109./30     ,    -1387./360    ,      1901./720    ,          0       ,          0       ,          0       ,           0       ,       0,        0],\
                         [    -95./288     ,      959./480     ,   -3649./720    ,     4991./720    ,     -2641./480    ,     4277./1440   ,          0       ,          0       ,           0       ,       0,        0],\
                         [  19087./60480   ,    -5603./2520    ,  135713./20160  ,   -10754./945    ,    235183./20160  ,   -18637./2520   ,   198721./60480  ,          0       ,           0       ,       0,        0],\
                         [  -5257./17280   ,    32863./13440   , -115747./13440  ,  2102243./120960 ,   -296053./13440  ,   242653./13440  , -1152169./120960 ,    16083./4480   ,           0       ,       0,        0],\
                         [1070017./3628800 , -4832053./1814400 ,19416743./1814400,-45586321./1814400,    862303./22680  ,-69927631./1814400, 47738393./1814400,-21562603./1814400,  14097247./3628800,       0,        0],\
                         [ -25713./89600   , 20884811./7257600 ,-2357683./181440 , 15788639./453600 ,-222386081./3628800,269181919./3628800,-28416361./453600 ,  6648317./181440 ,-104995189./7257600,4325321./1036800,0]], dtype=float)

    if (n<1):
        print("Der Eingabeparameter n muss mindestens 1 sein.")
        input()    # Erzwinge Interaktion mit Nutzer, damit er die Fehlermeldung sieht

        # exit, wenn gewuenscht
        sys.exit(1)
        
    elif (n > 10):
        print("Es sind nur die Koeffizienten der AB-Verfahren bis n=10 hinterlegt.")

        input()    # Erzwinge Interaktion mit Nutzer, damit er die Fehlermeldung sieht

        # exit, wenn gewuenscht
        
    A = Alpha_AB[0:n, 0:n+1]
    B = Beta_AB[0:n, 0:n+1]
    
    return A, B

def expLMM(f, T, x0, k, A, B, h, method='adams'):
    x_res = np.array([x0])
    t_res = [T[0]]
    cur_t = T[0]
    memory = np.array([f(x0, T[0])]) # memory of the k latest fi

    # starting phase
    for i in range(1, k):
        alpha = A[i-1, :]
        beta = B[i-1, :]

        # calculate the dot product of beta and the memory
        sum_beta = np.zeros(shape=(np.shape(x0)[0]))
        for j in range(i):
            sum_beta = sum_beta + beta[j] * memory[j, :]
        sum_beta = sum_beta * h
        
        # calculate the dot product of alpha and past results
        sum_alpha = np.zeros(shape=(np.shape(x0)[0]))
        for j in range(i):
            sum_alpha = sum_alpha + alpha[j] * x_res[j, :]

        # calculate xi+1
        cur_x = 1./alpha[i] * (sum_beta - sum_alpha)

        x_res = np.append(x_res, [cur_x], axis=0)
        cur_t = t_res[-1]+h
        t_res += [cur_t]
        memory = np.append(memory, [f(cur_x, cur_t)], axis=0)

    # computing phase
    alpha = A[k-1, :]
    beta = B[k-1, :]
    while cur_t < T[1]:
        # calculate the dot product of beta and the memory
        sum_beta = np.zeros(shape=(np.shape(x0)[0]))
        for j in range(k):
            sum_beta = sum_beta + beta[j] * memory[j, :]
        sum_beta = sum_beta * h

        # calculate the dot product of alpha and past results
        sum_alpha = np.zeros(shape=(np.shape(x0)[0]))
        for j in range(k):
            sum_alpha = sum_alpha + alpha[j] * x_res[j-k, :]

        # calculate xi+1
        cur_x = 1./alpha[k] * (sum_beta - sum_alpha)

        # update values
        x_res = np.append(x_res, [cur_x], axis=0)
        cur_t = t_res[-1]+h
        t_res += [cur_t]
        memory = np.append(memory, [f(cur_x, cur_t)], axis=0)
        memory = np.delete(memory, 0, 0)

    return x_res, t_res
