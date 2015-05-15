#!/usr/bin/env python
import unittest
import inverse
import numpy as np

tol = 1e-10
# make sure that nonlin_lstsq works for a linear case

def f_nonunique(model,x):
  out = 0.0*x
  for n,m in enumerate(model):
    out += m*np.sin(x/(1.0*(n+1)))*np.exp(-0.5*(n+1)*x)
  return out

M = 10
N = 50
seed = 4
np.random.seed(seed)
x = np.linspace(0,10.0,N)
model_true = np.linspace(2.0,1.0,M)
data = f_nonunique(model_true,x)
sigma = 0.01*np.ones(N)
noise = np.random.normal(0,0.01,N)
data += noise
reg = inverse.tikhonov_matrix(range(M),2)
penalty_range = np.power(10.0,np.linspace(-1,3,50))

class Test(unittest.TestCase):
  def test_CV(self):
    pred = inverse.GCV(penalty_range,f_nonunique,data,M,uncertainty=sigma,system_args=(x,),regularization=reg)
    best_pred_GCV = penalty_range[np.argmin(pred)]

    pred = inverse.CV(penalty_range,f_nonunique,data,M,uncertainty=sigma,system_args=(x,),regularization=reg)
    best_pred_CV = penalty_range[np.argmin(pred)]

    pred = inverse.KFCV(N,penalty_range,f_nonunique,data,M,uncertainty=sigma,system_args=(x,),regularization=reg)
    best_pred_KFCV = penalty_range[np.argmin(pred)]

    self.assertTrue(np.abs(best_pred_GCV - best_pred_CV) < tol)
    self.assertTrue(np.abs(best_pred_CV - best_pred_KFCV) < tol)

unittest.main()

'''
print('checking inverse.tikhonov for zeroth order')
reg = inverse.tikhonov_matrix([0,1,2],0)
assert np.linalg.norm(reg - np.eye(3)) < tol
print('passed')

print('checking inverse.tikhonov for first order')
reg = inverse.tikhonov_matrix([0,1,2],1)
assert np.linalg.norm(reg - np.array([[-1.0,1.0,0.0],[0,-1.0,1.0]])) < tol, 'failed'
print('passed')

print('checking inverse.tikhonov for second order')
reg = inverse.tikhonov_matrix([0,1,2],2)
assert np.linalg.norm(reg - np.array([[-1.0,1.0,0.0],
                                      [1.0,-2.0,1.0],
                                      [0.0,1.0,-1.0]])) < tol, 'failed'
print('passed')


'''


