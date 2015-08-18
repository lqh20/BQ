# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:32:14 2015

wsabi testing

@author: leonard
"""
import numpy as np
import wsabi
import bq_multidim as bq

alpha = 1e-5

test_1 = lambda x,y: np.exp(-x**2-y**2)+alpha

print bq.bayesian_quad_normal(test_1,2,200,mu=np.array([0,0]),Sigma=np.identity(2))
est,sd,model = wsabi.wsabi_mc(test_1,2,200,mu=np.array([0,0]),Sigma=np.identity(2))

est_list = []
sd_list = []
for i in range(20):
    est,sd,model = wsabi.wsabi_mc(test_1,2,100,mu=np.array([0,0]),Sigma=np.identity(2))
    est_list.append(est)
    sd_list.append(sd)
wsabi_mc(test_1,2,100,mu=np.array([0,0]),Sigma=np.identity(2))