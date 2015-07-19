# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:29:00 2015

implementation that works for arbitrary dimensions.

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



def bayesian_quad_lesbegue(f,n_dim,ranges, sample_size, kernel=None,random_samples= True, verbose = True):
    """
    Implements BMC/fixed grid Bayesian Quadrature for Lebesgue measure.
    
    Args:
        f (function): Function to be integrated
        n_dim (int): dimension
        ranges (list of lists of 2 floats): integration bounds
        sample_size (int): number of samples to use
    
    Kwargs:
        kernel (GPy kernel): kernel to use. Default is RBF with ARD
        random_samples (bool): whether to use random samples (BMC) or a fixed grid.
        verbose (bool): verbosity
    
    Returns:
        integral estimate, standard deviation
    """ 
    start = datetime.datetime.now()  
    if verbose:
        print start
    if kernel is None:
        kernel =GPy.kern.RBF(n_dim,ARD=True)
    #generate sample points
    assert n_dim == len(ranges)
    if verbose:
        print "sampling"
    if random_samples:
        x_grid_iter = zip(*[uniform(bounds[0],bounds[1]-bounds[0]).rvs(sample_size) for bounds in ranges])
        x_grid = np.array(x_grid_iter)
    else:
        x_grid_iter = []
        for i in range(sample_size):
            ppt = np.ravel(np.array([uniform(bounds[0],bounds[1]-bounds[0]).rvs(1) for bounds in ranges]))
            x_grid_iter.append(ppt)
        x_grid = np.array(x_grid_iter)
    #calculate function values
    f_obs = np.array(map(lambda x: f(*x),x_grid_iter))
       
    #fit GP model
    if verbose:
        print "done",datetime.datetime.now(), datetime.datetime.now()-start
        print "fitting model"
    model = gp_reg.GPRegression(x_grid,f_obs[:,None],kernel,normalizer=False)
    model["Gaussian*."].constrain_bounded(0,1e-10)
    model.optimize_restarts(1,verbose=verbose,optimizer="lbfgs",max_f_eval=1000)
    if verbose:
        print model
        print model.kern[:]
        print datetime.datetime.now(), datetime.datetime.now()-start
        print "integrals estimate"
    if isinstance(kernel,GPy.kern.RBF):
        print "RBF kernel detected"
        # set up functions that integrate kernel explicitly
        temp_func = lambda y,l,ranges: l*np.sqrt(2*np.pi)*(norm.cdf((ranges[1]-y)/l)-norm.cdf((ranges[0]-y)/l))
        funclist = [partial(temp_func,l=a[0],ranges=a[1]) for a in zip(model.kern["lengthscale"],ranges)]
        #set up the required product        
        tmp_prod = lambda y: model.kern["variance"]*np.prod([f(x) for f,x in zip(funclist,y)])
        int_kxX= np.ravel(np.array(map(tmp_prod,x_grid)))
        print "Mean integral computed", datetime.datetime.now(), datetime.datetime.now()-start
        print "Variance integral"
                
        def integral(l,ranges):
            x = (ranges[1]-ranges[0])/l
            return l**2*np.sqrt(2*np.pi)*(x*(norm.cdf(x)-norm.cdf(-x))+2*norm.pdf(x)-2*norm.pdf(0))
        
        int_kxx = model.kern["variance"]*np.prod(map(lambda x: integral(*x),zip(model.kern["lengthscale"],ranges)))
        int_kxx = int_kxx[0]
        #int_kxx_t, err = integrate.nquad(lambda *x: tmp_prod(x),ranges)
        #print int_kxx,int_kxx_t
    else:        
        int_kxX = np.array(map(lambda y: integrate.nquad( lambda *x : model.kern.K(np.atleast_2d(x),np.atleast_2d(y)),ranges)[0],x_grid))
        if verbose:
            print "Mean integral computed", datetime.datetime.now(), datetime.datetime.now()-start
            print "Variance integral"        
        int_kxx, err_kxx = integrate.nquad(lambda *x: model.kern.K(np.atleast_2d(x[:n_dim]),np.atleast_2d(x[n_dim:])),ranges+ranges)
    if verbose:
        print "initial integrals computed", datetime.datetime.now(), datetime.datetime.now()-start
        print "calculate estimate"
    integral_est = np.dot(int_kxX,LA.solve(model.kern.K(x_grid),f_obs))
    if verbose:
        print "estimate calculated", datetime.datetime.now(), datetime.datetime.now()-start
        print "calculate variance"
    variance = int_kxx-np.dot(int_kxX,LA.solve(model.kern.K(x_grid),int_kxX))
    if verbose:
        print "done", datetime.datetime.now(), datetime.datetime.now()-start
    # return mean estimate and standard deviation
    return integral_est, np.sqrt(variance)

def bayesian_quad_normal(f,n_dim, sample_size, mu, Sigma, verbose = True):
    """
    Implements BMC Bayesian Quadrature for a Gaussian base measure and RBF kernel.
    
    Args:
        f (function): Function to be integrated
        n_dim (int): dimension
        sample_size (int): number of samples to use
        mu (ndarray): mean vector of Gaussian to integrate against
        Sigma (ndarray): covariance matrix of Gaussian to integrate against
    
    Kwargs:
        verbose (bool): verbosity
    
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
       
    #fit GP model
    if verbose:
        print "done",datetime.datetime.now(), datetime.datetime.now()-start
        print "fitting model"
    model = gp_reg.GPRegression(x_grid,f_obs[:,None],kernel,normalizer=False)
    model["Gaussian*."].constrain_bounded(0,1e-10)
    model.optimize_restarts(verbose=verbose,optimizer="lbfgs",max_f_eval=1000)
    if verbose:
        print model
        print model.kern[:]
        print datetime.datetime.now(), datetime.datetime.now()-start
        print "integrals estimate"
    
    # calculate determinants
    DInv_vec = model.kern["lengthscale"]**(-2)
    #print DInv_vec
    DInv_mat = np.diag(DInv_vec)
    SigmaInv = LA.inv(Sigma)
    DInv_SigmaInv = DInv_mat + SigmaInv
    #print DInv_SigmaInv
    DInv_SigmaInv_Inv = LA.inv(DInv_SigmaInv)
    #print DInv_SigmaInv_Inv
    det_Sigma = LA.det(Sigma)
    det_DInv_SigmaInv = LA.det(DInv_SigmaInv)
    #print det_DInv_SigmaInv
        
    # set up functions that integrate kernel explicitly
    def int_k_x(y):
        y_prime = y-mu
        DInv_y_prime = np.einsum("i,i->i",y_prime,DInv_vec)
        temp = -0.5*np.einsum("i,i",DInv_y_prime,y_prime)+0.5*np.einsum("i,ij,j",DInv_y_prime,DInv_SigmaInv_Inv,DInv_y_prime)
        return model.kern["variance"]*det_Sigma**(-0.5)*det_DInv_SigmaInv**(-0.5)*np.exp(temp)
        
    int_kxX= np.ravel(np.array(map(int_k_x,x_grid)))
    #print int_kxX
    print "Mean integral computed", datetime.datetime.now(), datetime.datetime.now()-start
    print "Variance integral"
            
    def int_k_x_y():
        return model.kern["variance"]*det_Sigma**(-0.5)*LA.det(SigmaInv+2*DInv_mat)**(-0.5)
    
    int_kxx =  int_k_x_y()
    #print int_kxx
    #int_kxx_t, err = integrate.nquad(lambda *x: tmp_prod(x),ranges)
    #print int_kxx,int_kxx_t
    if verbose:
        print "initial integrals computed", datetime.datetime.now(), datetime.datetime.now()-start
        print "calculate estimate"
    integral_est = np.dot(int_kxX,LA.solve(model.kern.K(x_grid),f_obs))
    if verbose:
        print "estimate calculated", datetime.datetime.now(), datetime.datetime.now()-start
        print "calculate variance"
    variance = int_kxx-np.dot(int_kxX,LA.solve(model.kern.K(x_grid),int_kxX))
    #print variance
    if verbose:
        print "done", datetime.datetime.now(), datetime.datetime.now()-start
    # return mean estimate and standard deviation
    return integral_est, np.sqrt(variance)


def bayesian_quad_normal_embedding(f,n_dim, sample_size, mu, Sigma, emb_dim = 2, A=None, verbose = True):
    """
    Implements BMC Bayesian Quadrature with embedding for a Gaussian base measure and RBF kernel.
    
    Args:
        f (function): Function to be integrated
        n_dim (int): dimension
        sample_size (int): number of samples to use
        mu (ndarray): mean vector of Gaussian to integrate against
        Sigma (ndarray): covariance matrix of Gaussian to integrate against
    
    Kwargs:
        verbose (bool): verbosity
        A (ndarray): embedding to be used, defaults to random normal
    
    Returns:
        integral estimate, standard deviation
    """    
    start = datetime.datetime.now()  
    if verbose:
        print start
    kernel =GPy.kern.RBF(emb_dim,ARD=True)
    if A is None:
        A = np.random.normal(0,1,n_dim*emb_dim).reshape(emb_dim,n_dim)
    print A
    mu = np.atleast_1d(mu)
    Sigma = np.atleast_2d(Sigma)
    
    
    #generate sample points
    if verbose:
        print "sampling"
        x_grid = np.atleast_2d(multivariate_normal.rvs(mean=mu, cov=Sigma, size=sample_size))
    #calculate function values
    f_obs = np.array(map(lambda x: f(*x),np.atleast_2d(x_grid)))
       
    #fit GP model
    if verbose:
        print "done",datetime.datetime.now(), datetime.datetime.now()-start
        print "fitting model"
    
    x_grid_emb = np.dot(x_grid,A.transpose())
    model = gp_reg.GPRegression(x_grid_emb,f_obs[:,None],kernel,normalizer=False)
    model["Gaussian*."].constrain_bounded(0,1e-10)
    model.optimize_restarts(verbose=verbose,optimizer="lbfgs",max_f_eval=1000)
    if verbose:
        print model
        print model.kern[:]
        print datetime.datetime.now(), datetime.datetime.now()-start
        print "integrals estimate"
    
    # calculate determinants
    DInv_vec = model.kern["lengthscale"]**(-2)
    #print DInv_vec
    ADInvA = np.dot(np.dot(A.transpose(),np.diag(DInv_vec)),A)
    SigmaInv = LA.inv(Sigma)
    ADInvA_SigmaInv = ADInvA + SigmaInv
    #print DInv_SigmaInv
    ADInvA_SigmaInv_Inv = LA.inv(ADInvA_SigmaInv)
    #print DInv_SigmaInv_Inv
    det_Sigma = LA.det(Sigma)
    det_ADInvA_SigmaInv = LA.det(ADInvA_SigmaInv)
    #print det_DInv_SigmaInv
        
    # set up functions that integrate kernel explicitly
    def int_k_x(y):
        y_prime = y-mu
        ADInvA_y_prime = np.dot(ADInvA,y_prime)
        temp = -0.5*np.einsum("i,i",ADInvA_y_prime,y_prime)+0.5*np.einsum("i,ij,j",ADInvA_y_prime,ADInvA_SigmaInv_Inv,ADInvA_y_prime)
        return model.kern["variance"]*det_Sigma**(-0.5)*det_ADInvA_SigmaInv**(-0.5)*np.exp(temp)
        
    int_kxX= np.ravel(np.array(map(int_k_x,x_grid)))
    #print int_kxX
    print "Mean integral computed", datetime.datetime.now(), datetime.datetime.now()-start
    print "Variance integral"
            
    def int_k_x_y():
        return model.kern["variance"]*det_Sigma**(-0.5)*LA.det(SigmaInv+2*ADInvA)**(-0.5)
    
    int_kxx =  int_k_x_y()
    #print int_kxx
    #int_kxx_t, err = integrate.nquad(lambda *x: tmp_prod(x),ranges)
    #print int_kxx,int_kxx_t
    if verbose:
        print "initial integrals computed", datetime.datetime.now(), datetime.datetime.now()-start
        print "calculate estimate"
    integral_est = np.dot(int_kxX,LA.solve(model.kern.K(x_grid_emb),f_obs))
    if verbose:
        print "estimate calculated", datetime.datetime.now(), datetime.datetime.now()-start
        print "calculate variance"
    variance = int_kxx-np.dot(int_kxX,LA.solve(model.kern.K(x_grid_emb),int_kxX))
    #print variance
    if verbose:
        print "done", datetime.datetime.now(), datetime.datetime.now()-start
    # return mean estimate and standard deviation
    return integral_est, np.sqrt(variance)


