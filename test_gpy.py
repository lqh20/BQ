# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:54:23 2015

@author: leonard
"""

import sympy as sp
import numpy as np
import GPy

k = GPy.kern.RBF(2)

X = np.reshape(np.arange(16),(8,2))
Y = np.zeros((10,2))
print k.K(X,Y)
print k.K(X,Y).shape