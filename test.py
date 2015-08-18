# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:18:19 2015

@author: leonard
"""
import numpy as np
import GPy

N = 500
np.random.seed(1)
k = GPy.kern.Mahalanobis(2)
x = np.arange(N)/(N+0.1)
y = np.atleast_2d(x**2).T
y = y
X = np.zeros((N,2))
X[:,0] = x
X[:,1] = np.random.random(N)

print "X", X
print "y", y

model = GPy.models.GPRegression(X,y,k)
model["Gaussian"].constrain_bounded(0,1e-8)
model.optimize_restarts(100)
print model
print model.kern.A_matrix
print model.kern.weights()

N = 100
np.random.seed(1)
k = GPy.kern.Mahalanobis(1)
x = np.arange(N)
y = np.sin(np.atleast_2d(x).T)
X = np.atleast_2d(x).T

print "X", X
print "y", y

model = GPy.models.GPRegression(X,y,k)
model.optimize()

print model

model2 = GPy.models.GPRegression(X,y)
model2.optimize()

print model2

k2 = GPy.kern.RBF(1)

k2.K(X)
k.K(X)