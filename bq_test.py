# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:50:27 2015

1D BQ test with fixed grid and example function from position paper

@author: leonard
"""

import GPy
import GPy.models.gp_regression as gp_reg
import numpy as np
import numpy.linalg as LA
import scipy.integrate as integrate
import pylab as pb

pb.ion()

def f(x):
    return np.exp(-np.sin(3*x)**2-x**2)


def bayesian_quad_simple(f,sample_sizes, kernel):
    estimates = []
    variances = []
    for i in sample_sizes:
        x_grid = np.arange(-3.,3.,6./(i))[:,None]
        f_obs = f(x_grid)
           
        model = gp_reg.GPRegression(x_grid,f_obs,kernel,normalizer=False)
        model["Gaussian*."].constrain_bounded(0,1e-10)
        model.optimize_restarts(20,verbose=False,optimizer="lbfgs",max_f_eval=1000)
        
        int_kxX = np.array(map(lambda y: integrate.quad( lambda x : model.kern.K(np.atleast_2d(x),np.atleast_2d(y)),-3.,3.,limit =100)[0],x_grid))
        
        integral_est = np.dot(int_kxX,LA.solve(model.kern.K(x_grid),f_obs))
        
        int_kxx, err_kxx = integrate.dblquad(lambda x,y: model.kern.K(np.atleast_2d(x),np.atleast_2d(y)),-3.,3.,lambda x:-3.,lambda x:3.)
        
        variance = int_kxx-np.dot(int_kxX,LA.solve(model.kern.K(x_grid),int_kxX))
         #                    model.kern.K(x_grid,grid)))*grid_width**2).sum()
        
        # save
        estimates.append(integral_est)
        variances.append(variance)
        if (integral_est < 1) or (integral_est > 1.5):
            print kernel            
            print i
            print model
    
    return estimates,variances

k_rbf = GPy.kern.RBF(1,variance=0.05,lengthscale=0.25)
k_rbf["lengthscale"].constrain_bounded(0.001,0.33)
k_brown = GPy.kern.Brownian(1,variance=.05)
k_brown["variance"].constrain_bounded(0.0,10)
k_matern32 = GPy.kern.Matern32(1,variance=0.05,lengthscale=0.25)
k_matern32["lengthscale"].constrain_bounded(0.001,0.33)
k_matern52 = GPy.kern.Matern52(1,variance=0.05,lengthscale=0.25)
k_matern52["lengthscale"].constrain_bounded(0.001,0.33)

sample_sizes = [10,20,30,40,50,80,120,160,200]


rbf = bayesian_quad_simple(f,sample_sizes,k_rbf)
brown = bayesian_quad_simple(f,sample_sizes,k_brown)
matern32 = bayesian_quad_simple(f,sample_sizes,k_matern32)
matern52 = bayesian_quad_simple(f,sample_sizes,k_matern52)

# first plot

brown_plot = pb.plot(sample_sizes,brown[0],"b")
matern32_plot = pb.plot(sample_sizes,matern32[0],"g")
matern52_plot = pb.plot(sample_sizes,matern52[0],"y")
rbf_plot = pb.plot(sample_sizes,rbf[0],"r")
pb.title("Convergence with sample size")
pb.legend([brown_plot,matern32_plot,matern52_plot,rbf_plot],loc=0,labels=["Brownian","Matern 3/2","Matern 5/2","RBF"])

# log plot 
truth, err = integrate.quad(f,-3,3)
def transform(estimates):
    return np.abs(np.array(estimates)-truth)
brown_plot = pb.plot(sample_sizes,transform(brown[0]),"b")
matern32_plot = pb.plot(sample_sizes,transform(matern32[0]),"g")
matern52_plot = pb.plot(sample_sizes,transform(matern52[0]),"y")
rbf_plot = pb.plot(sample_sizes,transform(rbf[0]),"r")
pb.title("Convergence with sample size (absolute distance, log-scale)")
pb.legend([brown_plot,matern32_plot,matern52_plot,rbf_plot],loc=0,labels=["Brownian","Matern 3/2","Matern 5/2","RBF"])
pb.yscale("log")