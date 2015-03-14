#!/usr/bin/env python
import time as timemod
import datetime
import os
import numpy as np
import logging
from functools import wraps
logger = logging.getLogger(__name__)

##------------------------------------------------------------------------------
def decyear_inv(decyear,format='%Y-%m-%dT%H:%M:%S'):
  year = int(np.floor(decyear))
  remainder = decyear - year
  year_start = datetime.datetime(year,1,1)
  year_end = datetime.datetime(year+1,1,1)
  days_in_year = (year_end - year_start).days
  decdays = remainder*days_in_year
  date = year_start + datetime.timedelta(days=decdays)
  return date.strftime(format)

##------------------------------------------------------------------------------
def decyear(*args):
  date_tuple      = datetime.datetime(*args).timetuple()
  time_in_sec     = timemod.mktime(date_tuple)
  date_tuple      = datetime.datetime(args[0],1,1,0,0).timetuple()
  time_year_start = timemod.mktime(date_tuple)
  date_tuple      = datetime.datetime(args[0]+1,1,1,0,0).timetuple()
  time_year_end   = timemod.mktime(date_tuple)
  decimal_time    = (args[0] + (time_in_sec - time_year_start)
                     /(time_year_end - time_year_start))
  return decimal_time

##------------------------------------------------------------------------------
class Timer:
  def __init__(self):
    self.time_dict = {'init':timemod.time()}
    self.last = 'init'

  def tic(self,ID='process'):
    self.time_dict[ID] = timemod.time()
    self.last = ID

  def toc(self,ID=None):
    if ID is None:
      ID = self.last
    curtime = timemod.time()
    runtime = curtime - self.time_dict[ID]
    unit = 's'
    conversion = 1.0
    if runtime < 1.0:
      unit = 'ms'
      conversion = 1000.0
    if runtime > 60.0:
      unit = 'min'
      conversion = 1.0/60.0
    if runtime > 3600.0:
      unit = 'hr'
      conversion = 1.0/3600.0
    disp_runtime = '%.4g %s' % (runtime*conversion,unit)
    return 'elapsed time for %s: %s' % (ID,disp_runtime)

##------------------------------------------------------------------------------
def funtime(fun):
  '''
  decorator which times a function
  '''
  @wraps(fun)
  def subfun(*args,**kwargs):
    logger.info('evaluating %s' % fun.__name__)
    t = Timer()
    t.tic(fun.__name__)
    out = fun(*args,**kwargs)
    logger.info(t.toc(fun.__name__))
    return out
  return subfun

##------------------------------------------------------------------------------
def baseN_to_base10(value_baseN,N):
  base_char = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
  assert len(base_char) >= N
  value_baseN = str(value_baseN)
  base_char = base_char[:N]
  assert all(i in base_char for i in value_baseN)
  value_base10 = sum(base_char.find(i)*N**(n) for (n,i) in enumerate(value_baseN[::-1]))
  return value_base10  

##------------------------------------------------------------------------------
def base10_to_baseN(value_base10,N):
  base_char = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
  assert len(base_char) >= N
  value_baseN = ""
  while value_base10 != 0:
    value_baseN = base_char[value_base10%N] + value_baseN
    value_base10 = value_base10//N
  if len(value_baseN) == 0:
    value_baseN = base_char[0]
  return value_baseN

##------------------------------------------------------------------------------
def baseN_to_baseM(value_baseN,N,M):
  value_base10 = baseN_to_base10(value_baseN,N)
  value_baseM = base10_to_baseN(value_base10,M)
  return value_baseM
  
##------------------------------------------------------------------------------
def timestamp(factor=1.0):
  '''
  Description:
  Returns base 36 value of output from time.time() plus a character
  that identifies the computer this function was called from
  '''
  value_base10 = int(timemod.time()*factor)
  return baseN_to_baseM(value_base10,10,36)

##------------------------------------------------------------------------------
def list_flatten(lst):
  lst = list(lst)
  out = []
  for sub in lst:
    if hasattr(sub,'__iter__'):
      out.extend(list_flatten(sub))
    else:
      out.append(sub)
  return np.array(out)

##------------------------------------------------------------------------------
def divide_list(lst,N):
  '''                               
  Splits a list into N groups as evenly as possible         
  '''
  if len(lst) < N:
    N = len(lst)
  out = [[] for i in range(N)]
  for itr,l in enumerate(lst):
    out[itr%N] += [l]
  return out

##------------------------------------------------------------------------------
def rotation3D(argZ,argY,argX):
  '''                                
  creates a matrix which rotates a coordinate in 3 dimensional space about the 
  z axis by argz, the y axis by argy, and the x axis by argx, in that order  
  '''
  R1 = np.array([[  np.cos(argZ), -np.sin(argZ),           0.0],
                 [  np.sin(argZ),  np.cos(argZ),           0.0],
                 [           0.0,           0.0,           1.0]])

  R2 = np.array([[  np.cos(argY),           0.0,  np.sin(argY)],
                 [           0.0,           1.0,           0.0],
                 [ -np.sin(argY),           0.0,  np.cos(argY)]])

  R3 = np.array([[           1.0,           0.0,           0.0],
                 [           0.0,  np.cos(argX), -np.sin(argX)],
                 [           0.0,  np.sin(argX),  np.cos(argX)]])
  return R1.dot(R2.dot(R3))

##------------------------------------------------------------------------------
def find_indices(domain,realizations):
  '''  
  returns an array of indices such that domain[indices[n]] == realizations[n]
  '''
  if not hasattr(domain,'index'):
    domain = list(domain)

  if not hasattr(realizations,'index'):
    realizations = list(realizations)
  
  domain_set = set(domain)
  realizations_set = set(realizations)
  if len(domain_set) != len(domain):
      raise ValueError('domain contains repeated values')

  if not realizations_set.issubset(domain_set):
    intersection = realizations_set.intersection(domain_set)
    not_in_domain = realizations_set.difference(intersection)
    for i in not_in_domain:
      raise ValueError('item %s not found in domain' % i)

  indices = [domain.index(r) for r in realizations]
  return indices

##------------------------------------------------------------------------------
def pad(M,pad_shape,value=0,dtype=None):
  '''
  returns an array containing the values from M but the ends of each dimension
  are padded with 'value' so that the returned array has shape 'pad_shape'
  '''
  M = np.array(M)
  M_shape = np.shape(M)

  assert len(M_shape) == len(pad_shape), ('new_shape must have the same '
         'number of dimensions as M') 
  assert all([m <= n for m,n in zip(M_shape,pad_shape)]), ('length of each new '
         'dimension must be greater than or equal to the corresponding '
         'dimension of M')

  if dtype is None:
    dtype = M.dtype

  out = np.empty(pad_shape,dtype=dtype)
  out[...] = value
 
  if not all(M_shape):
    return out

  M_dimension_ranges = [range(m) for m in M_shape]
  out[np.ix_(*M_dimension_ranges)] = M

  return out

##------------------------------------------------------------------------------
def pad_stack(arrays,axis=0,**kwargs):
  '''
  stacks array along the specified dimensions and any inconsistent dimension
  sizes are dealt with by padding the smaller array with 'value'  
  '''
  array_shapes = [np.shape(a) for a in arrays]
  array_shapes = np.array(array_shapes)
  pad_shape = np.max(array_shapes,0)
  padded_arrays = []
  for a in arrays:
    a_shape = np.shape(a)
    pad_shape[axis] = a_shape[axis]     
    padded_arrays += [pad(a,pad_shape,**kwargs)]
  
  out = np.concatenate(padded_arrays,axis=axis)

  return out
    


      
