#!/usr/bin/env python
import copy
import sys
import os
import numpy as np
import logging
from nonlin_lstsq import nonlin_lstsq
from nonlin_lstsq import jacobian_fd
import misc
logger = logging.getLogger(__name__)

def _parser(args,kwargs):
  '''
  parses args and kwargs that are used for nonlin_lstsq
  ''' 
  data_no = len(args[1])
  if type(args[2]) == int:
    m_no = np.ones(args[2])
  else:
    m_no = len(args[2])
  arguments = {'system':args[0],
               'data':args[1],
               'm_o':args[2],
               'data_no':data_no,
               'm_no':m_no,
               'regularization':kwargs.get('regularization'),
               'sigma':kwargs.get('sigma',np.ones(data_no)),
               'system_args':kwargs.get('system_args',()),
               'system_kwargs':kwargs.get('system_kwargs',{}),
               'jacobian':kwargs.get('jacobian',jacobian_fd),
               'jacobian_args':kwargs.get('jacobian_args',()),
               'jacobian_kwargs':kwargs.get('jacobian_kwargs',{})}

  if arguments['jacobian'] == jacobian_fd:
    arguments['jacobian_args'] = (arguments['system'],)
    arguments['jacobian_kwargs'] = {'system_args':arguments['system_args'],
                                    'system_kwargs':arguments['system_kwargs']}    
  return arguments

def CV(penalty_range,*args,**kwargs):
  '''
  Leave-one-out cross validation method for finding the optimal penalty 
  parameter which scales the regularization matrix used in nonlin_lstsq.  

  This function takes the same arguments as nonlin_lstsq with the additional 
  penalty_range argument.  One of the provided key word arguments must be 
  'regularization'.
  
  PARAMETERS
  ----------
    penalty_range: array of penalty parameters to test
    *args: arguments for nonlin_lstsq
    **kwargs: key word arguments for nonlin_lstsq

  RETURNS
  -------
    L2 norm of the residuals for each penalty parameter.  The residuals consist
    of the difference between each set of test data and their prediction.

  reference: Aster et al. 2005, Parameter Estimation and Inverse Problems 

  '''
  L2_list = np.zeros(len(penalty_range))
  regularization = kwargs.pop('regularization')
  for itr,p in enumerate(penalty_range):
    scaled_regularization = p*regularization
    L2_list[itr] = LOOCV_step(*args,
                              regularization=scaled_regularization,
                              **kwargs)
  return L2_list

def GCV(penalty_range,*args,**kwargs):
  '''
  General cross validation method for finding the optimal penalty 
  parameter which scales the regularization matrix used in nonlin_lstsq.  

  This function takes the same arguments as nonlin_lstsq with the additional 
  penalty_range argument.  One of the provided key word arguments must be 
  'regularization'.
  
  PARAMETERS
  ----------
    penalty_range: array of penalty parameters to test
    *args: arguments for nonlin_lstsq
    **kwargs: key word arguments for nonlin_lstsq

  RETURNS
  -------
    L2 norm of the residuals for each penalty parameter.  The residuals consist
    of the difference between each set of test data and their prediction.

  reference: Aster et al. 2005, Parameter Estimation and Inverse Problems 

  '''
  L2_list = np.zeros(len(penalty_range))
  regularization = kwargs.pop('regularization')
  for itr,p in enumerate(penalty_range):
    scaled_regularization = p*regularization
    L2_list[itr] = GCV_step(*args,
                            regularization=scaled_regularization,
                            **kwargs)
  return L2_list

def KFCV(K,penalty_range,*args,**kwargs):
  '''
  K-folds cross validation method for finding the optimal penalty 
  parameter which scales the regularization matrix used in nonlin_lstsq.  

  This function takes the same arguments as nonlin_lstsq with the additional 
  penalty_range argument.  One of the provided key word arguments must be 
  'regularization'.
  
  As opposed to the leave-one-out cross validation, this algorithm successively 
  leaves out one of K groups of data.  The data groups are randomly determines
  using np.random.choice
  
  PARAMETERS
  ----------
    K: number of folds
    penalty_range: array of penalty parameters to test
    *args: arguments for nonlin_lstsq
    **kwargs: key word arguments for nonlin_lstsq

  RETURNS
  -------
    L2 norm of the residuals for each penalty parameter.  The residuals consist
    of the difference between each set of test data and their prediction.

  reference: Aster et al. 2005, Parameter Estimation and Inverse Problems 

  '''
  parsed_args = _parser(args,kwargs)
  L2_list = np.zeros(len(penalty_range))
  regularization = kwargs.pop('regularization')
  groups = misc.divide_list(np.random.choice(parsed_args['data_no'],
                                             parsed_args['data_no'],
                                             replace=False),K)
  for itr,p in enumerate(penalty_range):
    scaled_regularization = p*regularization
    L2_list[itr] = KFCV_step(groups,*args,
                             regularization=scaled_regularization,
                             **kwargs)
  return L2_list

def LOOCV_step(*args,**kwargs):
  parsed_args = _parser(args,kwargs)
  residual = np.zeros(parsed_args['data_no'])
  # loop over data indices to leave out
  for idx in range(parsed_args['data_no']):
    # data_indices consists of all data indices which are not 
    # left out of the inversion
    data_indices = range(parsed_args['data_no'])
    data_indices.remove(idx)
    # predicted model parameters with excluded data
    m_pred = nonlin_lstsq(parsed_args['system'],
                          parsed_args['data'],
                          parsed_args['m_o'],
                          data_indices=data_indices,
                          **kwargs)
    # predicted data
    data_pred = parsed_args['system'](m_pred,
                                      *parsed_args['system_args'],
                                      **parsed_args['system_kwargs'])
    # difference between predicted data and the exluded data
    residual[idx] = data_pred[idx] - parsed_args['data'][idx]
    residual[idx] /= parsed_args['sigma'][idx] 
  # return L2 norm of residuals
  L2 = residual.dot(residual)
  return L2

def GCV_step(*args,**kwargs):
  parsed_args = _parser(args,kwargs)
  # predicted model parameters with provided penalty parameter
  m_pred = nonlin_lstsq(parsed_args['system'],
                        parsed_args['data'],
                        parsed_args['m_o'],
                        **kwargs)
  # predicted data
  data_pred = parsed_args['system'](m_pred,
                                    *parsed_args['system_args'],
                                    **parsed_args['system_kwargs'])
  # predicted data, normalized by uncertainty
  residual = (data_pred - parsed_args['data'])
  residual /= parsed_args['sigma']
  # Jacobian of the last iteration in nonlin_lstsq.  If the system 
  # was linear w.r.t the unknowns then this is the system matrix.
  jac = parsed_args['jacobian'](m_pred,
                                *parsed_args['jacobian_args'],
                                **parsed_args['jacobian_kwargs'])

  # weight the jacobian by the data uncertainty
  jac = (1.0/parsed_args['sigma'])[:,None]*jac
  L = parsed_args['regularization']
  # compute the generalized inverse 
  jac_inv = np.linalg.inv(jac.transpose().dot(jac) + 
                          L.transpose().dot(L)).dot(jac.transpose())
  # this is evaluating the formula from Aster et. al 2005.
  # There is the one notable difference that I am NOT normalizing by the 
  # number of data points
  num = parsed_args['data_no']**2*(residual.dot(residual))
  den = np.trace(np.eye(parsed_args['data_no']) - jac.dot(jac_inv))**2
  return num/den

def KFCV_step(groups,*args,**kwargs):
  parsed_args = _parser(args,kwargs)
  residual = np.zeros(parsed_args['data_no'])
  for indices in groups:
    # find data indices which are not in 'indices'
    data_indices = [i for i in range(parsed_args['data_no']) if not i in indices]
    m_pred = nonlin_lstsq(*args,
                          data_indices=data_indices,
                          **kwargs)
    data_pred = parsed_args['system'](m_pred,
                                      *parsed_args['system_args'],
                                      **parsed_args['system_kwargs'])
    residual[indices] = data_pred[indices] - parsed_args['data'][indices]
    residual[indices] /= parsed_args['sigma'][indices] 
  L2 = residual.dot(residual)
  return L2

##------------------------------------------------------------------------------
def cross_validate(exclude_groups,*args,**kwargs):
  '''
  BEING PHASED OUT

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
  L2 = np.linalg.norm(residual)
  return L2



