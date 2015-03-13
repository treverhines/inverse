#!/usr/bin/env python
import numpy as np
import logging
logger = logging.getLogger(__name__)

def correlated_noise(var,decay,times):
  N = len(times)
  mean = np.zeros(N)
  t1,t2 = np.meshgrid(times,times)
  cov = var*np.exp(-np.abs(t1 - t2)/decay)
  noise = np.random.multivariate_normal(mean,cov,1)
  return noise[0]
