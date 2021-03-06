#!/usr/bin/env python
import copy
import numpy as np
import logging
from misc import funtime
from misc import list_flatten

logger = logging.getLogger(__name__)

##------------------------------------------------------------------------------
def _remove_zero_rows(M):
  '''
  used in tikhonov_matrix
  '''
  return np.array(filter(np.any,M))

##------------------------------------------------------------------------------
def _linear_to_array_index(val,shape):
  '''
  used in next method of IndexEnumerate
  '''
  N = len(shape)
  indices = np.zeros(N,int)
  for count,dimsize in enumerate(shape[::-1]):
    indices[N-(count+1)] = val%dimsize
    val = val//dimsize
  return indices

##------------------------------------------------------------------------------
class Perturb:
  def __init__(self,v,delta=1):
    self.v = v
    self.delta = delta
    self.itr = 0 
    self.N = len(v)

  def __iter__(self):
    return self

  def next(self):
    if self.itr == self.N:
      raise StopIteration
    else:
      out = copy.deepcopy(self.v)
      out[self.itr] += self.delta
      self.itr += 1
      return out
      

##------------------------------------------------------------------------------
class IndexEnumerate:
  def __init__(self,C):
    '''
    used in tikhonov matrix

    enumerates over the flattened elements of C and their index locations in C
  
    e.g.
  
    >> C = np.array([[1,2],[3,4]])
    >> for idx,val in IndexEnumerate(C):
    ...  print('idx: %s, val: %s' % (idx,val))   
    ...
    idx: [0, 0], val: 1
    idx: [0, 1], val: 2
    idx: [1, 0], val: 3
    idx: [1, 1], val: 4
    '''
    self.C = np.asarray(C)
    self.shape = np.shape(C)
    self.size = np.size(C)
    self.itr = 0

  def __iter__(self):
    return self

  def next(self):
    if self.itr == self.size:
      raise StopIteration
    else:
      idx = _linear_to_array_index(self.itr,self.shape)
      self.itr += 1
      return (idx,self.C[tuple(idx)])

##------------------------------------------------------------------------------
class Neighbors(IndexEnumerate):
  '''
  Iterator that Loops over elements in array C returning the element and its  
  neighbors
  '''
  def __init__(self,C,search='all'):
    IndexEnumerate.__init__(self,C)
    assert search in ['all','forward','backward']
    self.search = search

  def next(self):
    idx,val = IndexEnumerate.next(self)
    neighbors = np.zeros(0,dtype=int)
    if (self.search == 'all') | (self.search == 'forward'):
      for idx_pert in Perturb(idx,1):
        if any(idx_pert >= self.shape):
          continue
        neighbors = np.append(neighbors,self.C[tuple(idx_pert)])

    if (self.search == 'all') | (self.search == 'backward'):
      for idx_pert in Perturb(idx,-1):
        if any(idx_pert < 0):
          continue
        neighbors = np.append(neighbors,self.C[tuple(idx_pert)])
    
    return neighbors,val

##------------------------------------------------------------------------------
def _tikhonov_zeroth_order(C,L):
  '''
  used in tikhonov_matrix
  '''
  for val in C.flat:
    if (val == -1):
      continue
    L[val,val] = 1

  return L

##------------------------------------------------------------------------------
def _tikhonov_first_order(C,L):
  '''
  used in tikhonov_matrix
  '''
  shape = np.shape(C)
  Lrow = 0
  for neighbors,i in Neighbors(C,'forward'):
    if i == -1:
      continue
    for k in neighbors:
      if k == -1:
        continue
      L[Lrow,i] += -1
      L[Lrow,k] += 1
      Lrow += 1

  return L 

##------------------------------------------------------------------------------
def _tikhonov_second_order(C,L):
  '''
  used in tikhonov_matrix
  '''
  shape = np.shape(C)
  for neighbors,i in Neighbors(C,'all'):
    if i == -1:
      continue
    order = sum(neighbors != -1)
    for k in neighbors:
      if k == -1:
        continue
      L[i,i] += -1.0/order  
      L[i,k]  +=  1.0/order

  return L

## Tikhonov matrix
##------------------------------------------------------------------------------
@funtime
def tikhonov_matrix(C,n,column_no=None,dtype=None):
  '''
  Parameters
  ----------
    C: connectivity matrix, this can contain '-1' elements which can be used
       to break connections. 
    n: order of tikhonov regularization
    column_no: number of columns in the output matrix
    sparse_type: either 'bsr','coo','csc','csr','dia','dok','lil'

  Returns
  -------
    L: tikhonov regularization matrix saved as a csr sparse matrix

  Example
  -------
    first order regularization for 4 model parameters which are related in 2D 
    space
      >> Connectivity = [[0,1],[2,3]]
      >> L = tikhonov_matrix(Connectivity,1)
      
  '''     
  C = np.array(C)
  # check to make sure all values (except -1) are unique
  idx = C != -1
  params = C[idx] 
  unique_params = set(params)
  assert len(params) == len(unique_params), (
         'all values in C, except for -1, must be unique')

  Cdim = len(np.shape(C))
  max_param = np.max(C) + 1
  if column_no is None:
    column_no = max_param

  assert column_no >= max_param, (
         'column_no must be at least as large as max(C)')

  if n == 0:
    L = np.zeros((column_no,column_no),dtype=dtype)
    L =  _tikhonov_zeroth_order(C,L)

  if n == 1:
    L = np.zeros((Cdim*column_no,column_no),dtype=dtype)
    L = _tikhonov_first_order(C,L)

  if n == 2:
    L = np.zeros((column_no,column_no),dtype=dtype)
    L = _tikhonov_second_order(C,L)

  L = _remove_zero_rows(L)     

  return L


