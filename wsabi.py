# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:28:05 2015

WSABI implementation

@author: leonard
"""
import GPy
import GPy.models.gp_regression as gp_reg
import numpy as np
import numpy.linalg as LA
import scipy.integrate as integrate
from scipy.stats import uniform
from scipy.stats import norm
from scipy.stats import multivariate_normal
import datetime
from functools import partial


def wsabi_mc(f,n_dim, sample_size, mu, Sigma, verbose = True, alpha = 8e-6):
    """
    Implements WSABI-MC for a Gaussian base measure and RBF kernel.
    
    Args:
        f (function): Function to be integrated
        n_dim (int): dimension
        sample_size (int): number of samples to use
        mu (ndarray): mean vector of Gaussian to integrate against
        Sigma (ndarray): covariance matrix of Gaussian to integrate against
    
    Kwargs:
        verbose (bool): verbosity
        alpha (float): parameter of WSABI
    Returns:
        integral estimate, standard deviation
    """    
    start = datetime.datetime.now()  
    if verbose:
        print start
    kernel =GPy.kern.RBF(n_dim,ARD=True)
    mu = np.atleast_1d(mu)
    Sigma = np.atleast_2d(Sigma)
    
    #generate sample points
    if verbose:
        print "sampling"
    x_grid = np.atleast_2d(multivariate_normal.rvs(mean=mu, cov=Sigma, size=sample_size))
    #calculate function values
    f_obs = np.array(map(lambda x: f(*x),np.atleast_2d(x_grid)))
    
    f_tilde_obs = np.sqrt(2*(f_obs-alpha))
    
    #fit GP model
    if verbose:
        print "done",datetime.datetime.now(), datetime.datetime.now()-start
        print "fitting model"
    model = gp_reg.GPRegression(x_grid,f_tilde_obs[:,None],kernel,normalizer=False)
    model["Gaussian*."].constrain_bounded(0,1e-10)
    model.optimize_restarts(verbose=verbose,optimizer="lbfgs",max_f_eval=1000)
    if verbose:
        print model
        print "parameters",model.kern[:]
        print datetime.datetime.now(), datetime.datetime.now()-start
        print "integrals estimate"
    
    # calculate determinants
    DInv_vec = model.kern["lengthscale"]**(-2)
    #print DInv_vec
    DInv_mat = np.diag(DInv_vec)
    SigmaInv = LA.inv(Sigma)
    DInv_SigmaInv = DInv_mat + SigmaInv
    D2Inv_SigmaInv = DInv_mat + DInv_SigmaInv
    D3Inv_SigmaInv = DInv_mat + D2Inv_SigmaInv
    #print DInv_SigmaInv
    D2Inv_SigmaInv_Inv = LA.inv(D2Inv_SigmaInv)
    #print DInv_SigmaInv_Inv
    det_Sigma = LA.det(Sigma)
    det_DInv_SigmaInv = LA.det(DInv_SigmaInv)
    det_D2Inv_SigmaInv = LA.det(D2Inv_SigmaInv)
    det_D3Inv_SigmaInv = LA.det(D3Inv_SigmaInv)
    #print det_DInv_SigmaInv
    #pre compute 
    Sm = np.dot(SigmaInv,mu)
    mSm = np.dot(mu,Sm)
    DX = np.einsum("j,ij->ij",DInv_vec,x_grid)
    DXip = np.expand_dims(DX,1)+np.expand_dims(DX,0) 
    XDX = np.einsum("in,in-> i",x_grid,DX)
    XDXip = np.expand_dims(XDX,1)+np.expand_dims(XDX,0)
        
    # set up functions that integrate kernel explicitly
    def int_k_x():
        v = DXip+Sm
        temp = -0.5*(XDXip+mSm-np.einsum("ipn,nm,ipm->ip",v,D2Inv_SigmaInv_Inv,v))
        return model.kern["variance"]**2*det_Sigma**(-0.5)*det_D2Inv_SigmaInv**(-0.5)*np.exp(temp)
        
    int_kx = int_k_x()
    print int_kx
    print "Mean integral computed", datetime.datetime.now(), datetime.datetime.now()-start
    print "Variance integral"
            
    def int_k_x_y():
        A = D2Inv_SigmaInv
        B = -DInv_mat
        AInv = D2Inv_SigmaInv_Inv
        A_hat = LA.inv(A-np.dot(B,np.dot(AInv,B)))
        B_hat = - np.dot(AInv,np.dot(B,A_hat))
        #inversion of block matrices
        v1 = np.expand_dims(Sm,0)+DX
        vAv = np.einsum("in,ni->i",v1,LA.solve(A-np.dot(B,np.dot(AInv,B)),v1.transpose()))
        temp = -0.5*(2*mSm+XDXip+np.expand_dims(vAv,0)+np.expand_dims(vAv,1)+2*np.einsum("in,nm,pm->ip",v1,B_hat,v1))
        return model.kern["variance"]**3*det_Sigma**(-0.5)*det_D3Inv_SigmaInv**(-0.5)*np.exp(temp)
    
    int_kxy =  int_k_x_y()
    #print int_kxx
    #int_kxx_t, err = integrate.nquad(lambda *x: tmp_prod(x),ranges)
    #print int_kxx,int_kxx_t
    if verbose:
        print "initial integrals computed", datetime.datetime.now(), datetime.datetime.now()-start
        print "calculate estimate"
    Kl = LA.solve(model.kern.K(x_grid),f_tilde_obs)
    integral_est = alpha+0.5*np.einsum("ij,i,j",int_kx,Kl,Kl)
    if verbose:
        print "estimate calculated", datetime.datetime.now(), datetime.datetime.now()-start
        print "calculate variance"
        
    variance = np.einsum("ij,i,j",int_kxy,Kl,Kl)-np.einsum("ip,pm,i,m",int_kx,LA.solve(model.kern.K(x_grid),int_kx),Kl,Kl)
    #print variance
    if verbose:
        print "done", datetime.datetime.now(), datetime.datetime.now()-start
    # return mean estimate and standard deviation
    return integral_est, np.sqrt(variance), model