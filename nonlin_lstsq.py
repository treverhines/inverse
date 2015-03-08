#!/usr/bin/env python
import copy
import sys
import os
import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import logging
from tikhonov import Perturb
from misc import list_flatten

logger = logging.getLogger(__name__)

##------------------------------------------------------------------------------
class Converger:
  def __init__(self,final,atol=0.01,rtol=0.01,norm=2):
    self.atol = atol
    self.rtol = rtol
    self.norm = norm
    self.final = np.asarray(final)
    self.L2 = np.inf
    return

  def __call__(self,current):
    current = np.asarray(current)
    L2_new = np.linalg.norm(current - self.final,self.norm)    
    if not np.isfinite(L2_new):
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

##------------------------------------------------------------------------------
def jacobian_fd(m_o,
                system,
                system_args=None,
                system_kwargs=None,
                dm=0.01,
                dtype=None):
  '''
  Parameters
  ----------
    system: function where the first argument is a list of model parameters and 
            the output is a data list
    m_o: location in model space where the jacobian will be computed. must be a
         mutable sequence (e.g. np.array or list)
    system_args: additional arguments to system
    system_kargs: additional key word arguments to system
    dm: step size used for the finite difference approximation

  Returns
  -------
    J:  jacobian matrix with dimensions: len(data),len(parameters)
  ''' 
  if system_args is None:
    system_args = []
  if system_kwargs is None:
    system_kwargs = {}

  data_o         = system(m_o,*system_args,**system_kwargs)
  param_no       = len(m_o)
  data_no        = len(data_o)
  Jac            = np.zeros((data_no,param_no),dtype=dtype)
  i = 0
  for m_pert in Perturb(m_o,dm):
    data_pert = system(m_pert,*system_args,**system_kwargs)
    Jac[:,i]  = (data_pert - data_o)/dm
    i += 1

  return Jac

##------------------------------------------------------------------------------
def _objective(system,
               data,
               sigma,
               system_args,
               system_kwargs,
               jacobian,
               jacobian_args,
               jacobian_kwargs,
               reg_matrix,
               lm_matrix,
               data_indices):
  '''
  used for nonlin_lstsq
  '''  
  data = np.asarray(data)
  sigma = np.asarray(sigma)
  def objective_function(model):
    '''
    evaluates the function to be minimized for the given model
    '''
    pred = system(model,*system_args,**system_kwargs)
    res = (pred - data) / sigma
    res = res[data_indices]
    reg = reg_matrix.dot(model)    
    lm = np.zeros(np.shape(lm_matrix)[0])
    return np.hstack((res,reg,lm))

  def objective_jacobian(model):
    '''
    evaluates the jacobian of the objective function at the given model
    '''
    jac = jacobian(model,*jacobian_args,**jacobian_kwargs)
    jac = jac / sigma[:,np.newaxis]
    jac = jac[data_indices,:]
    return np.vstack((jac,reg_matrix,lm_matrix))

  return objective_function,objective_jacobian

##------------------------------------------------------------------------------
def lstsq(G,d,*args,**kwargs):
  '''
  used by nonlin_lstsq
  '''
  out = np.linalg.lstsq(G,d,*args,**kwargs)[0]
  return out

##------------------------------------------------------------------------------
def nnls(G,d,*args,**kwargs):
  '''
  used by nonlin_lstsq
  '''
  out = scipy.optimize.nnls(G,d)[0]
  return out

##------------------------------------------------------------------------------
def nonlin_lstsq(system,
                 data,
                 m_o,
                 sigma=None,
                 system_args=None,
                 system_kwargs=None,
                 jacobian=None,
                 jacobian_args=None,
                 jacobian_kwargs=None,
                 solver=lstsq,
                 solver_args=None,
                 solver_kwargs=None,
                 reg_matrix=None,
                 LM_damping=False,
                 LM_param=10.0,
                 LM_factor=2.0,   
                 maxitr=20,
                 rtol=1.0e-2,
                 atol=1.0e-2,
                 data_indices=None,
                 dtype=None):
  '''
  Newtons method for solving a least squares problem

  PARAMETERS
  ----------
  *args 
  -----
    system: function where the first argument is a vector of model parameters 
            and the remaining arguments are system args and system kwargs
    data: vector of data values
    m_o: vector of model parameter initial guesses

  **kwargs 
  -------
    system_args: list of arguments to be passed to system following the model
                 parameters
    system_kwargs: list of key word arguments to be passed to system following 
                   the model parameters
    jacobian: function which computes the jacobian w.r.t the model parameters.
              the first arguments is a vector of parameters and the remaining 
              arguments are jacobian_args and jacobian_kwargs
    jacobian_args: arguments to be passed to the jacobian function 
    jacobian_kwargs: key word arguments to be passed to the jacobian function 
    solver: function which solves "G*m = d" for m, where the first two arguments
            are G and d.  inverse.lstsq, and inverse.nnls are wrappers for 
            np.linalg.lstsq, and scipy.optimize.nnls and can be used here. Using
            nnls ensures that the output model parameters are non-negative
    solver_args: additional arguments for the solver after G and d
    solver_kwargs: additional key word arguments for the solver 
    sigma: data uncertainty vector
    reg_matrix: regularization matrix scaled by the penalty parameter
    LM_damping: flag indicating whether to use the Levenberg Marquart algorithm 
                which damps step sizes in each iteration but ensures convergence
    LM_param: starting value for the Levenberg Marquart parameter 
    LM_factor: the levenberg-Marquart parameter is either multiplied or divided
               by this value depending on whether the algorithm is converging or
               diverging. 
    maxitr: number of steps for the inversion
    rtol: Algorithm stops if relative L2 between successive iterations is below 
          this value  
    atol: Algorithm stops if absolute L2 is below this value 
    data_indices: indices of data that will be used in the inversion. Defaults 
                  to using all data.

  Returns
  -------
    m_new: best fit model parameters
  '''
  param_no = len(m_o)
  data_no = len(data)

  m_o = np.array(m_o,dtype=dtype)
  data = np.array(data,dtype=dtype)

  if sigma is None:
    sigma = np.ones(data_no,dtype=dtype)

  if system_args is None:
    system_args = []

  if system_kwargs is None:
    system_kwargs = {}

  if jacobian is None:
    jacobian = jacobian_fd
    jacobian_args = [system]
    jacobian_kwargs = {'system_args':system_args,
                       'system_kwargs':system_kwargs,
                       'dtype':dtype}

  if jacobian_args is None:
    jacobian_args = []

  if jacobian_kwargs is None:
    jacobian_kwargs = {}

  if solver_args is None:
    solver_args = []

  if solver_kwargs is None:
    solver_kwargs = {}

  if data_indices is None:
    data_indices = range(data_no)

  if reg_matrix is None:
    reg_matrix = np.zeros((0,param_no),dtype=dtype)

  if hasattr(reg_matrix,'todense'):
    reg_matrix = np.array(reg_matrix.todense())

  if LM_damping:
    lm_matrix = LM_param*np.eye(param_no,dtype=dtype)
  else:
    lm_matrix = np.zeros((0,param_no),dtype=dtype)
 
  obj_func,obj_jac = _objective(system,
                                data,
                                sigma,
                                system_args,
                                system_kwargs,
                                jacobian,
                                jacobian_args,
                                jacobian_kwargs,
                                reg_matrix,
                                lm_matrix,
                                data_indices)

  final = np.zeros(data_no+
                   np.shape(reg_matrix)[0] +
                   np.shape(lm_matrix)[0])

  conv = Converger(final,atol=atol,rtol=rtol)
  count = 0
  status = None
  while not ((status == 0) | (status == 3) | (count == maxitr)):
    J = obj_jac(m_o)
    J = np.asarray(J,dtype=dtype)
    d = obj_func(m_o)
    d = np.asarray(d,dtype=dtype)
    m_new = solver(J,-d+J.dot(m_o))
    d_new = obj_func(m_new)
    status,message = conv(d_new)
    logger.debug(message)
    if (status == 1) and LM_damping:
      logger.debug('decreasing LM parameter to %s' % LM_param)
      lm_matrix /= LM_factor
      LM_param /= LM_factor

    while ((status == 2) | (status == 3)) and LM_damping:
      logger.debug('increasing LM parameter to %s' % LM_param)
      lm_matrix *= LM_factor
      LM_param *= LM_factor
      J = obj_jac(m_o)
      J = np.asarray(J,dtype=dtype)
      d = obj_func(m_o)
      d = np.asarray(d,dtype=dtype)
      m_new = solver(J,-d+J.dot(m_o))
      d_new = obj_func(m_new)
      status,message = conv(d_new)
      logger.debug(message)

    m_o = m_new
    conv.set(d_new)
    count += 1
    if count == maxitr:
      logger.debug('converged due to maxitr')

  return m_o


