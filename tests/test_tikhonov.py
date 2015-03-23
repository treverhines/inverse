#!/usr/bin/env python
import unittest
import inverse
import numpy as np

tol = 1e-10
# make sure that nonlin_lstsq works for a linear case

class Test(unittest.TestCase):
  def test_zeroth_order(self):
    reg = inverse.tikhonov_matrix([0,1,2],0)
    self.assertTrue(np.linalg.norm(reg - np.eye(3)) < tol)

  def test_first_order(self):
    reg = inverse.tikhonov_matrix([0,1,2],1)
    self.assertTrue(np.linalg.norm(reg - np.array([[-1.0,1.0,0.0],[0,-1.0,1.0]])) < tol)

  def test_second_order(self):
    reg = inverse.tikhonov_matrix([0,1,2],2)
    self.assertTrue(np.linalg.norm(reg - np.array([[-1.0,1.0,0.0],
                                                   [0.5,-1.0,0.5],
                                                   [0.0,1.0,-1.0]])) < tol)

unittest.main()



