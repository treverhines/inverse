#!/usr/bin/env python
import numpy as np
import logging
from nllstsq import nonlin_lstsq
from inverse_misc import list_flatten
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

