# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:18:15 2015

@author: leonard
"""
import GPy
import numpy as np
import bq_multidim as bq
import pylab as pb
import scipy.integrate as integrate



pb.ion()

def f(x,y):
    return np.exp(-np.sin(3*x)**2-y**2)
    
def g(x1,x2,x3,x4,x5):
    return 10* np.sin(np.pi*x1*x2) + 20*(x3 - 0.5)**2 + 10*x4 + 5*x5
    


print bq.bayesian_quad(f,2,[[-1,1],[-1,1]],200), integrate.nquad(f,[[-1,1],[-1,1]])
print bq.bayesian_quad(g,5,[[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]],500), integrate.nquad(g,[[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]])

print bq.bayesian_quad(f,2,[[-1,1],[-1,1]],200,kernel=GPy.kern.Matern32(2))

k_rbf = GPy.kern.RBF(1,variance=0.05,lengthscale=0.25)
k_rbf["lengthscale"].constrain_bounded(0.001,0.33)
k_brown = GPy.kern.Brownian(1,variance=.05)
k_brown["variance"].constrain_bounded(0.0,10)
k_matern32 = GPy.kern.Matern32(1,variance=0.05,lengthscale=0.25)
k_matern32["lengthscale"].constrain_bounded(0.001,0.33)
k_matern52 = GPy.kern.Matern52(1,variance=0.05,lengthscale=0.25)
k_matern52["lengthscale"].constrain_bounded(0.001,0.33)




A = np.random.normal(0,1,10).reshape(2,5)
print bq.bayesian_quad(g,5,[[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]],500), integrate.nquad(g,[[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]])

def test_1(x,y):
    return x**2+y**2
    
def test_2(*x):
    x = np.array(x)
    return 1
    
def test_10(*x):
    x = np.array(x)
    return (x**2).sum()

def test_emb(*x):
    x = np.array(x)
    return (x**2)[:2].sum()

mu = np.array([0,0])
Sigma=np.diag([1,1])
sample_size = 10 
x_grid = np.atleast_2d(multivariate_normal.rvs(mean=mu, cov=Sigma, size=sample_size))

bq.bayesian_quad_normal(test_1,2,500,mu = np.array([0,0]),Sigma=np.diag([1,1]), verbose=False)

bq.bayesian_quad_normal(test_2,2,10,mu = np.array([0,0]),Sigma=np.diag([1,1]))


bq.bayesian_quad_normal(test_10,10,100,mu = np.repeat(0,10),Sigma = np.diag(np.repeat(1,10)))
bq.bayesian_quad_normal_embedding(test_10,10,1000,mu = np.repeat(0,10),Sigma = np.diag(np.repeat(1,10)))

bq.bayesian_quad_normal_embedding(test_2,10,500,mu = np.repeat(0,10),Sigma = np.diag(np.repeat(1,10)))

A = np.zeros((2,10))
A[0,0] =1
A[1,1] = 1

bq.bayesian_quad_normal_embedding(test_emb,10,500,mu = np.repeat(0,10),Sigma = np.diag(np.repeat(1,10)),A=A)

est, sd, model = bq.bayesian_quad_normal_embedding_ml(test_emb,10,500,mu = np.repeat(0,10),Sigma = np.diag(np.repeat(1,10)),As=A)
