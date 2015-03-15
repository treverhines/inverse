#!/usr/bin/env python
import numpy as np

class Converger:
  def __init__(self,final,atol=0.01,rtol=0.01,maxitr=100,norm=2):
    self.atol = atol
    self.rtol = rtol
    self.maxitr = maxitr
    self.norm = norm
    self.final = np.asarray(final)
    self.L2 = np.inf
    self.itr = 0

  def __call__(self,current):
    self.itr += 1
    current = np.asarray(current)
    L2_new = np.linalg.norm(current - self.final,self.norm)
    if self.itr >= self.maxitr:
      message = 'finished due to maxitr'
      return 0,message

    elif not np.isfinite(L2_new):
      message = 'encountered invalid L2'
      return 3,message

    elif L2_new <= self.atol:
      message = 'converged due to atol:          L2=%s' % L2_new
      return 0,message

    elif abs(L2_new - self.L2) <= self.rtol:
      message = 'converged due to rtol:          L2=%s' % L2_new
      return 0,message

    elif L2_new < self.L2:
      message = 'converging:                     L2=%s' % L2_new
      return 1,message

    elif (L2_new >= self.L2):
      message = 'diverging:                      L2=%s' % L2_new
      return 2,message

  def set(self,current):
    self.current = np.asarray(current)
    self.L2 = np.linalg.norm(current - self.final,self.norm)

    return
