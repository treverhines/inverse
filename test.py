#!/usr/bin/env python
from __future__ import division
import numpy as np

def _W(x,n,D,H):
  W1 = np.arctan((2*n*H+D)/x)
  W2 = np.arctan((2*n*H-D)/x)
  return 1.0/np.pi * (W1 - W2)

def _F(x,n,N):
  top = n/N
  bottom = (n+1)/N
  return 0.5*(_W(x,0,bottom,1.0) - _W(x,0,top,1.0))

def _G(x,n,N):
  top = 1.0 + 10*n/N
  bottom = 1.0 + 10*(n+1)/N
  return _W(x,1,1.0,top) - _W(x,1,1.0,bottom)

def test_forward(m,N=100,noise=None):
  '''
  large scale nolinear forward problem. Divides m divides into m1 and m2 and 
  then return 

    u(x_i) = m1_j*F_j(x_i) + m1_j*m2_k*G_jk(x_i),


  I will come up with a better test problem later
  '''
  out = np.zeros(N)
  x = np.linspace(0.1,20,N)
  m1 = m[:len(m)//2]
  m2 = m[len(m1):]
  m1N = len(m1)
  m2N = len(m2)
  for i,m1_ in enumerate(m1):
    out += m1_*_F(x,i,m1N) 
    for j,m2_ in enumerate(m2):
      out += m1_*m2_*_G(x,j,m2N)
    
  if noise is not None:
    out += np.random.normal(0,noise,N)

  return out
  
  
  
