# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:28:56 2015

Test GPy script

@author: leonard
"""
import pylab as pb
import GPy
import numpy as np


d= 1
var = 1.
theta = 0.2

k = GPy.kern.RBF(d,var,theta)

k.plot()

k = GPy.kern.RBF(d)

lengthscales = np.asarray([0.2,0.5,1.,2.,4.])

for t in lengthscales:
     k['.*lengthscale']=t
     k.plot()
     pb.legend(t)

variances = np.asarray([0.2,0.5,1,2,4])

for t in variances:
     k['.*variance']=t
     k.plot()
    
pb.legend(variances)

kb = GPy.kern.Brownian(input_dim=1)
kb.plot(x = 2.,plot_limits=[0,5])
kb.plot(x = 4.,plot_limits=[0,5],ls='--',color='r')
pb.ylim([-0.1,5.1])

# kernels are psd
k = GPy.kern.Matern52(input_dim=2)
X = np.random.rand(50,2)

C = k.K(X,X)
np.linalg.eigvals(C) 

# sample paths

k1 = GPy.kern.RBF(input_dim=1,lengthscale=0.1)
k2 = GPy.kern.Matern32(input_dim=1,variance=1.,lengthscale=0.1)
k3 = GPy.kern.Brownian(1)
X = np.linspace(0.,1.,500) # 500 points evenly spaced over [0,1]
X = X[:,None] # reshape X to make it n*D

mu = np.zeros((500)) # vector of the means
C1 = k1.K(X,X) # covariance matrix
C2 = k2.K(X,X) # covariance matrix
C3 = k3.K(X,X)

# Generate 20 sample path with mean mu and covariance C
Z1 = np.random.multivariate_normal(mu,C1,20)
Z2 = np.random.multivariate_normal(mu,C2,20)
Z3 = np.random.multivariate_normal(mu,C3,20)


pb.figure() # open new plotting window

for i in range(20):
    pb.plot(X[:],Z1[i,:])
    pb.title("RBF")

pb.figure()
for i in range(20):
    pb.plot(X[:],Z2[i,:])
    pb.title("Matern 3/2")

pb.figure()
for i in range(20):
    pb.plot(X[:],Z3[i,:])
    pb.title("Brownian motion")

# GP regression model

#generate data
X = np.linspace(0.05,0.95,10)[:,None]
Y = -np.cos(np.pi*X) +np.sin(4*np.pi*X) + np.random.randn(10,1)*0.05
pb.figure()
pb.plot(X,Y,'kx',mew=1.5)




