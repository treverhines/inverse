#!/usr/bin/env python
import copy
import sys
import os
import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import logging
from nonlin_lstsq import nonlin_lstsq
from misc import list_flatten

logger = logging.getLogger(__name__)

##------------------------------------------------------------------------------
def model_covariance(G,sigma,rcond=1e-15):
  '''
  Returns the model covariance matrix.
  
  Arguments:
    G: system matrix
    sigma: data uncertainty
    rcond: cut off value for pinv

  '''  
  sigma_inv = 1.0/sigma
  Gg = np.linalg.pinv(sigma_inv[:,None]*G,rcond)
  Cm = Gg.dot(Gg.transpose())
  return Cm

##------------------------------------------------------------------------------
def cross_validate(exclude_groups,*args,**kwargs):
  '''
  cross validation routine.  This function runs nonlin_lstsq to find the optimal
  model parameters while excluding each of the groups of data indices given in 
  exclude groups.  It returns the L2 norm of the predicted data for each of the
  excluded groups minus the observed data

  PARAMETERS
  ----------
    exclude_groups: list of groups of data indices to exclude
    *args: arguments for nonlin_lstsq
    **kwargs: arguments for nonlin_lstsq
  '''                     
  logger.info('starting cross validation iteration')
  system = args[0]
  system_args = kwargs.get('system_args',())
  system_kwargs = kwargs.get('system_kwargs',{})
  data = args[1]
  data_no = len(data)
  sigma = kwargs.get('sigma',np.ones(data_no))
  parameters = args[2]
  group_no = len(exclude_groups)
  param_no = len(parameters)
  residual = np.zeros(data_no)
  for itr,exclude_indices in enumerate(exclude_groups):
    data_indices = [i for i in range(data_no) if not i in exclude_indices]
    pred_params = nonlin_lstsq(*args,
                               data_indices=data_indices,
                               **kwargs)
    pred_data = system(pred_params,*system_args,**system_kwargs)
    residual[exclude_indices] = pred_data[exclude_indices] - data[exclude_indices]
    residual[exclude_indices] /= sigma[exclude_indices] # normalize residuals
    logger.info('finished cross validation for test group %s of '
                '%s' % (itr+1,group_no))
  L2 = np.linalg.norm(residual)
  logger.info('finished cross validation with predicted L2: %s' % L2)
  return L2

##------------------------------------------------------------------------------
def bootstrap(bootstrap_iterations,*args,**kwargs):
  '''
  Bootstraps the uncertainties of the best fit model parameters found by
  nonlin_lstsq
  
  Parameters
  ----------
    bootstrap_iterations: number of bootstrap iterations
    *args: arguments to be given to nonlin_lstsq
    **kwargs: key word arguments to be given to nonlin_lstsq
    bootstrap_log_level: controls the verbosity
  
  Returns
  -------
    parameter_array: array of best fit model parameters for each iteration
  ''' 
  logger.info('starting bootstrap')
  data = args[1]
  parameters = args[2]
  data_no = len(data)
  param_no = len(parameters)
  parameter_array = np.zeros((bootstrap_iterations,param_no))
  for i in range(bootstrap_iterations):
    data_indices = np.random.choice(range(data_no),data_no)
    pred_params = nonlin_lstsq(*args,
                               data_indices=data_indices,
                               **kwargs)
    parameter_array[i,:] = pred_params
    logger.info('finished bootstrap iteration %s of %s' 
                % ((i+1),bootstrap_iterations))
  return parameter_array

##------------------------------------------------------------------------------
def block_bootstrap(bootstrap_iterations,data_groups,*args,**kwargs):
  '''
  Bootstraps the uncertainties of the best fit model parameters found by
  nonlin_lstsq
  
  Parameters
  ----------
    iterations: number of bootstrap iterations
    data_groups: list of data groups where each data group is a list of 
      data indices within that group
    *args: arguments to be given to nonlin_lstsq
    **kwargs: key word arguments to be given to nonlin_lstsq
    bootstrap_log_level: controls the verbosity

  Returns
  -------
    parameter_array: array of best fit model parameters for each iteration
  ''' 
  logger.info('starting bootstrap')
  data = args[1]
  parameters = args[2]
  data_no = len(data)
  param_no = len(parameters)
  group_no = len(data_groups)
  parameter_array = np.zeros((bootstrap_iterations,param_no))
  for i in range(bootstrap_iterations):
    test_groups = np.random.choice(range(group_no),group_no)
    data_indices = list_flatten([data_groups[k] for k in test_groups]) 
    pred_params = nonlin_lstsq(*args,
                               data_indices=data_indices,
                               **kwargs)
    parameter_array[i,:] = pred_params
    logger.info('finished bootstrap iteration %s of %s' 
                 % ((i+1),bootstrap_iterations))
  return parameter_array

