#!/usr/bin/env python
'''
routines which solve min_m||Gm - d|| with potentially additional
constraints on m.
'''
import numpy as np
import scipy.optimize
import pymls
from converger import Converger
import logging
logger = logging.getLogger(__name__)

def _arg_checker(fin):
  def fout(G,d,*args,**kwargs):
    G = np.asarray(G)
    d = np.asarray(d)
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
  lower_lim = np.asarray(lower_lim)
  upper_lim = np.asarray(upper_lim)
  llim_shape = np.shape(lower_lim)
  ulim_shape = np.shape(upper_lim)
  G_shape = np.shape(G)
  assert llim_shape == (G_shape[1],), ('lower_lim must be a 1D vector with '
                                       'length equal to the number of model '
                                       'parameters')
  assert ulim_shape == (G_shape[1],), ('upper_lim must be a 1D vector with '
                                       'length equal to the number of model '
                                       'parameters')
  d = d[:,None]
  lower_lim = lower_lim[:,None]
  upper_lim = upper_lim[:,None]
  out = pymls.bounded_lsq(G,d,lower_lim,upper_lim)
  out = np.squeeze(out)
  return out

@_arg_checker
def cgls(G,d,m_o=None,maxitr=2000,rtol=1e-6,atol=1e-6):
  '''
  congugate gradient least squares

  algorithm from Aster et al. 2005
  '''
  N,M = np.shape(G)
  if m_o is None:
    m_o = np.zeros(M)

  s = d - G.dot(m_o)
  p = np.zeros(M)
  r = s.dot(G)
  r_prev = np.zeros(M)
  beta = 0

  conv = Converger(np.zeros(N),atol=atol,rtol=rtol,maxitr=maxitr)
  status,message = conv(s)
  logger.debug(message)
  k = 0
  while (status !=  0) & (status != 3):
    if k > 0:
      beta = r.dot(r)/r_prev.dot(r_prev)
    p = r + beta*p
    Gp = G.dot(p)
    alpha = r.dot(r)/Gp.dot(Gp)
    #print('hi')
    #alpha = r.dot(r)/Gp.dot(Gp)
    m_o = m_o + alpha*p
    s = s - alpha*Gp
    r_prev[:] = r
    r = s.dot(G)
    status,message = conv(s)
    conv.set(s)
    logger.debug(message)
    k += 1

  return m_o

class _Cgls(object):
  '''
  an instance of this class behaves like the function cgls except that 
  in the absence of an initial m_o, the m_o returned from the preivious
  call is used.  If the function was not called before then m_o is a 
  vector of zeros.  This is very useful for a nonlinear least squares 
  algorithm.
  '''
  def __init__(self):
    self.m_o = None

  def __call__(self,*args,**kwargs):    
    if self.m_o is None:
      self.m_o = cgls(*args,**kwargs)
    else:
      m_o = kwargs.pop('m_o',self.m_o)
      self.m_o = cgls(*args,m_o=m_o,**kwargs)
    return self.m_o

_cgls_instance = _Cgls()
@_arg_checker
def cgls_state(*args,**kwargs):
  '''
  conjugate gradient solver with state

  behaves exactly as cgls except if this function has been called 
  previously in the current python session then the default m_o is 
  changed to the output from the previous call. 
  
  The 'state' is stored in the callable instance '_cgls_instance'
  '''
  return _cgls_instance(*args,**kwargs)

