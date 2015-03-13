#!/usr/bin/env python
'''
routines which solve min_m||Gm - d|| with potentially additional
constraints on m.
'''
import numpy as np
import scipy
import pymls

def _arg_checker(fin):
  def fout(G,d,*args,**kwargs):
    G_shape = np.shape(G)  
    d_shape = np.shape(d)
    if len(d_shape) > 1:
      d = np.squeeze(d)    
      d_shape = np.shape(d)

    assert len(d_shape) == 1
    assert len(G_shape) == 2
    assert G_shape[0] == d_shape[0]
    output = fin(G,d,*args,**kwargs)
    assert len(output) == G_shape[1]
    return output

  fout.__doc__ = fin.__doc__
  fout.__name__ = fin.__name__
  return fout

@_arg_checker
def lstsq(G,d,*args,**kwargs):
  '''                                                                                                              
  wrapper from scipy.linalg.lstsq  
  '''
  out = scipy.linalg.lstsq(G,d,*args,**kwargs)[0]
  return out

@_arg_checker
def nnls(G,d,*args,**kwargs):
  '''                                                                                                              
  wrapper from scipy.optimize.nnls
  '''
  out = scipy.optimize.nnls(G,d,*args,**kwargs)[0]
  return out

@_arg_checker
def bounded_lstsq(G,d,lower_lim,upper_lim):
  '''
  wrapper for pymls.bounded_lsq

  finds m minimizes ||Gm - d|| subject to the constraint that

    lower_lim[i] < m[i] < upper_lim[i]

  Parameters
  ----------
    G: system matrix (N,M)
    d: data vector (N,)
    lower_lim: lower limit on m (M,)
    upper_lim: upper limit on m (M,)

  Returns
  -------
    best fit model vector with the applied constraints (M,)

  '''
  d = d[:,None]
  lower_lim = lower_lim[:,None]
  upper_lim = upper_lim[:,None]
  out = pymls.bounded_lsq(G,d,lower_lim,upper_lim)
  out = np.squeeze(out)
  return out


