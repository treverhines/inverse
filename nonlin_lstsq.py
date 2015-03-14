#!/usr/bin/env python
import numpy as np
import logging
import solvers
from tikhonov import Perturb
from tikhonov import tikhonov_matrix
from inverse_misc import funtime
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
def _residual(system,
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
  def residual_function(model):
    '''
    evaluates the function to be minimized for the given model
    '''
    pred = system(model,*system_args,**system_kwargs)
    res = (pred - data) / sigma
    res = res[data_indices]
    reg = reg_matrix.dot(model)    
    lm = np.zeros(np.shape(lm_matrix)[0])
    return np.hstack((res,reg,lm))

  def residual_jacobian(model):
    '''
    evaluates the jacobian of the objective function at the given model
    '''
    jac = jacobian(model,*jacobian_args,**jacobian_kwargs)
    jac = jac / sigma[:,np.newaxis]
    jac = jac[data_indices,:]
    return np.vstack((jac,reg_matrix,lm_matrix))

  return residual_function,residual_jacobian

def _arg_parser(args,kwargs):
  # define kwargs that do not default to None
  assert len(args) == 3, 'nonlin_lstsq takes exactly 3 positional arguments'

  p = {'solver':solvers.lstsq,
       'LM_damping':False,
       'LM_param':10.0,
       'LM_factor':2.0,
       'maxitr':20,
       'rtol':1.0e-2,
       'atol':1.0e-2,
       'sigma':None,
       'system_args':None,
       'system_kwargs':None,
       'jacobian':None,
       'jacobian_args':None,
       'jacobian_kwargs':None,
       'solver_args':None,
       'solver_kwargs':None,
       'data_indices':None,
       'regularization':None,
       'dtype':None}

  p.update(kwargs)
  p['system'] = args[0]
  p['data'] = args[1]  
  p['m_o'] = args[2]

  # if the initial guess is an integer, then interpret it as length of the model
  # parameter vector and assume ones as the initial guess
  if type(p['m_o']) == int:
    p['m_o'] = np.ones(p['m_o'])

  p['param_no'] = len(p['m_o'])
  p['data_no'] = len(p['data'])

  # if no uncertainty is given then assume it is ones.
  if p['sigma'] is None:
    p['sigma'] = np.ones(p['data_no'])

  if p['system_args'] is None:
    p['system_args'] = []

  if p['system_kwargs'] is None:
    p['system_kwargs'] = {}

  # if no jacobian is provided then set use the finite difference approximation
  if p['jacobian'] is None:
    p['jacobian'] = jacobian_fd
    p['jacobian_args'] = [p['system']]
    p['jacobian_kwargs'] = {'system_args':p['system_args'],
                            'system_kwargs':p['system_kwargs']}

  if p['jacobian_args'] is None:
    p['jacobian_args'] = []

  if p['jacobian_kwargs'] is None:
    p['jacobian_kwargs'] = {}

  if p['solver_args'] is None:
    p['solver_args'] = []

  if p['solver_kwargs'] is None:
    p['solver_kwargs'] = {}

  # default to assuming all data will be used.  This functionality is added for
  # to make cross validation easier
  if p['data_indices'] is None:
    p['data_indices'] = range(p['data_no'])

  # if regularization is a array or tuple of length 2 then assume it describes
  # the regularization order and the penalty parameter then create the
  # regularization matrix
  if np.shape(p['regularization'])==(2,):
    order = p['regularization'][0]
    mag = p['regularization'][1]
    p['regularization'] = mag*tikhonov_matrix(range(p['param_no']),order)

  if p['regularization'] is None:
    p['regularization'] = np.zeros((0,p['param_no']))

  # if regularization is given as a sparse matrix and unsparsify it
  if hasattr(p['regularization'],'todense'):
    p['regularization'] = np.array(p['regularization'].todense())

  if p['LM_damping']:
    p['lm_matrix'] = p['LM_param']*np.eye(p['param_no'])
  else:
    p['lm_matrix'] = np.zeros((0,p['param_no']))

  return p

##------------------------------------------------------------------------------
def nonlin_lstsq(*args,**kwargs):
  '''
  Newtons method for solving a least squares problem

  PARAMETERS
  ----------
  *args 
  -----
    system: function where the first argument is a vector of model parameters 
            and the remaining arguments are system args and system kwargs
    data: vector of data values (N,)
    m_o: vector of model parameter initial guesses.  If an integer is provided
         then the initial guess will be a vector of ones with that length (M,)  

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
    regularization: regularization matrix scaled by the penalty parameter.  This
                    is a (*,M) array.                              
                                         OR
                    array of length 2 where the first argument is the tikhonov 
                    regularization order and the second argument is the penalty
                    parameter.  The regularization matrix is assembled assuming
                    that the position of the model parameters in the vector m 
                    corresponds to their spatial relationship.

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
  p = _arg_parser(args,kwargs)
 
  res_func,res_jac = _residual(p['system'],
                               p['data'],
                               p['sigma'],
                               p['system_args'],
                               p['system_kwargs'],
                               p['jacobian'],
                               p['jacobian_args'],
                               p['jacobian_kwargs'],
                               p['regularization'],
                               p['lm_matrix'],
                               p['data_indices'])

  final = np.zeros(len(p['data_indices']) +
                   np.shape(p['regularization'])[0] +
                   np.shape(p['lm_matrix'])[0])

  conv = Converger(final,atol=p['atol'],rtol=p['rtol'])
  count = 0
  status = None
  while not ((status == 0) | (status == 3) | (count == p['maxitr'])):
    J = res_jac(p['m_o'])
    J = np.asarray(J,dtype=p['dtype'])
    d = res_func(p['m_o'])
    d = np.asarray(d,dtype=p['dtype'])
    m_new = p['solver'](J,-d+J.dot(p['m_o']),
                        *p['solver_args'],
                        **p['solver_kwargs'])
    d_new = res_func(m_new)
    status,message = conv(d_new)
    logger.debug(message)
    if (status == 1) and p['LM_damping']:
      logger.debug('decreasing LM parameter to %s' % p['LM_param'])
      p['lm_matrix'] /= p['LM_factor']
      p['LM_param'] /= p['LM_factor']

    while ((status == 2) | (status == 3)) and p['LM_damping']:
      logger.debug('increasing LM parameter to %s' % p['LM_param'])
      p['lm_matrix'] *= p['LM_factor']
      p['LM_param'] *= p['LM_factor']
      J = res_jac(p['m_o'])
      J = np.asarray(J,dtype=p['dtype'])
      d = res_func(m_o)
      d = np.asarray(d,dtype=p['dtype'])
      m_new = p['solver'](J,-d+J.dot(p['m_o']),
                          *p['solver_args'],
                          **p['solver_kwargs'])
      d_new = res_func(m_new)
      status,message = conv(d_new)
      logger.debug(message)

    p['m_o'] = m_new
    conv.set(d_new)
    count += 1
    if count == p['maxitr']:
      logger.debug('converged due to maxitr')

  return p['m_o']
