# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:23:17 2015

Mahalanobis kernel.

@author: leonard
"""

# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kern import Kern
from ...core.parameterization import Param
import numpy as np

class Mahalanobis(Kern):
    """
    Mahalanobis kernel

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance:
    :type variance: float
    :param A_matrix: embedding matrix
    :type A_matrix: numpy matrix
    """
    def __init__(self, input_dim, variance=1., A_matrix=None, active_dims=None, name='Mahalanobis'):
        super(Mahalanobis, self).__init__(input_dim, active_dims, name)
        if A_matrix is None:
            # non-zero, so gradient is non-zero
            A_matrix = np.identity(input_dim)+0.001*np.ones((input_dim,input_dim))
        self.variance = Param('variance', variance)
        #self.lengthscale = Param('lengthscale',lengthscale) not needed. implicit in A.
        self.A_matrix = Param('A_matrix',A_matrix)
        self.link_parameters(self.variance)
        self.link_parameters(self.A_matrix)
        self["variance"].constrain_positive()

    def K(self,X,X2=None):
        if X2 is None:
            X2 = X
        #AX = np.dot(X,self.A_matrix.T)
        #AX2 = np.dot(X2,self.A_matrix.T)
        #result = np.empty((AX.shape[0],AX2.shape[0]))
        #for i in range(AX.shape[0]):
        #    for j in range(i,AX2.shape[0]):
        #        result[i,j] = self.variance*np.exp(-0.5*((AX[i,:]-AX2[j,:])**2).sum())
        #        result[j,i] = result[i,j]
        X1_ext = np.einsum("iq,j->ijq",X,np.ones(X2.shape[0]))
        #print X1_ext.shape
        X2_ext = np.einsum("jq,i->ijq",X2,np.ones(X.shape[0]))
        #print X2_ext.shape
        result2 = self.variance*np.einsum("mij->ij",np.exp(-0.5*np.einsum("mn,ijn->mij",np.array(self.A_matrix),X1_ext-X2_ext)**2))
        #print np.allclose(result, result2)
        return result2

    def Kdiag(self,X):
        return self.variance
        
    def weights(self):
        return np.dot(np.array(self.A_matrix).transpose(),np.array(self.A_matrix))

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        #print "dL_dK", dL_dK
        self.variance.gradient = np.einsum("mn,mn->",dL_dK,self.K(X,X2)/self.variance)
        X1_ext = np.einsum("iq,j->ijq",X,np.ones(X.shape[0]))
        X2_ext = np.einsum("jq,i->ijq",X2,np.ones(X.shape[0]))
        #print X1_ext-X2_ext
        #print "dL_dK",dL_dK
        self.A_matrix.gradient = np.einsum("ij,ij,pq,ijq->pq",dL_dK,self.K(X,X2),-np.array(self.A_matrix),(X1_ext-X2_ext)**2)
        #print "A_gradient",self.A_matrix.gradient

    #def update_gradients_diag(self, dL_dKdiag, X):
        #self.variance.gradient = np.dot(np.abs(X.flatten()), dL_dKdiag)

    #def gradients_X(self, dL_dK, X, X2=None):
        #if X2 is None:
            #return np.sum(self.variance*dL_dK*np.abs(X),1)[:,None]
        #else:
            #return np.sum(np.where(np.logical_and(np.abs(X)<np.abs(X2.T), np.sign(X)==np.sign(X2)), self.variance*dL_dK,0.),1)[:,None]


class Mahalanobis_lengthscale(Kern):
    """
    Mahalanobis kernel

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance:
    :type variance: float
    :param A_matrix: embedding matrix
    :type A_matrix: numpy matrix
    """
    def __init__(self, input_dim, variance=1.,lengthscale=1., A_matrix=None, active_dims=None, name='Mahalanobis'):
        super(Mahalanobis, self).__init__(input_dim, active_dims, name)
        if A_matrix is None:
            # non-zero, so gradient is non-zero
            A_matrix = np.identity(input_dim)+0.001*np.ones((input_dim,input_dim))
        self.variance = Param('variance', variance)
        if len(np.array(lengthscale)) != A_matrix.shape[0]:
            self.lengthscale = Param('lengthscale',np.repeat(lengthscale,A_matrix.shape[0]) 
        else:
            self.lengthscale = Param('lengthscale',lengthscale) 
        
        self.A_matrix = Param('A_matrix',A_matrix)
        self.link_parameters(self.variance)
        self.link_parameters(self.A_matrix)
        self["variance"].constrain_positive()

    def K(self,X,X2=None):
        if X2 is None:
            X2 = X
        #AX = np.dot(X,self.A_matrix.T)
        #AX2 = np.dot(X2,self.A_matrix.T)
        #result = np.empty((AX.shape[0],AX2.shape[0]))
        #for i in range(AX.shape[0]):
        #    for j in range(i,AX2.shape[0]):
        #        result[i,j] = self.variance*np.exp(-0.5*((AX[i,:]-AX2[j,:])**2).sum())
        #        result[j,i] = result[i,j]
        X1_ext = np.einsum("iq,j->ijq",X,np.ones(X2.shape[0]))
        #print X1_ext.shape
        X2_ext = np.einsum("jq,i->ijq",X2,np.ones(X.shape[0]))
        #print X2_ext.shape
        result2 = self.variance*np.einsum("mij->ij",np.exp(-0.5*np.einsum("mn,ijn,m->mij",np.array(self.A_matrix),X1_ext-X2_ext)**2),1./np.array(self.lengthscale)**2)
        #print np.allclose(result, result2)
        return result2

    def Kdiag(self,X):
        return self.variance
        
    def weights(self):
        return np.dot(np.array(self.A_matrix).transpose(),np.array(self.A_matrix))

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        #print "dL_dK", dL_dK
        self.variance.gradient = np.einsum("mn,mn->",dL_dK,self.K(X,X2)/self.variance)
        X1_ext = np.einsum("iq,j->ijq",X,np.ones(X.shape[0]))
        X2_ext = np.einsum("jq,i->ijq",X2,np.ones(X.shape[0]))
        #print X1_ext-X2_ext
        #print "dL_dK",dL_dK
        self.A_matrix.gradient = np.einsum("ij,ij,pq,ijq,q->pq",dL_dK,self.K(X,X2),-np.array(self.A_matrix),(X1_ext-X2_ext)**2,1./np.array(self.lengthscale)**2)
        #print "A_gradient",self.A_matrix.gradient

    #def update_gradients_diag(self, dL_dKdiag, X):
        #self.variance.gradient = np.dot(np.abs(X.flatten()), dL_dKdiag)

    #def gradients_X(self, dL_dK, X, X2=None):
        #if X2 is None:
            #return np.sum(self.variance*dL_dK*np.abs(X),1)[:,None]
        #else:
            #return np.sum(np.where(np.logical_and(np.abs(X)<np.abs(X2.T), np.sign(X)==np.sign(X2)), self.variance*dL_dK,0.),1)[:,None]



