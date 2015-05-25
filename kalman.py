#!/usr/bin/env python
from __future__ import division
import numpy as np
from inverse import jacobian_fd
from inverse import nonlin_lstsq

class KalmanFilter:
  '''Uses an Extended Kalman filter to evaluate the state, x, 
  for each time step

  Attributes
  ----------

    transition: function which takes the state and a 'context'
      dictionary and returns a prediction of the next state as well as
      the uncertainty in that prediction. If no uncertainty is 
      specified then the uncertainty is assumed to be zero.

    observation: function which takes the state and a 'context'
      dictionary and returns the predicted observations.

    state: list of states from each iteration

    state_cov: list of state covariances from each iteration

    predicted: list of predicted observations from each iteration

    predicted_cov: list of predicted covariances from each
      iteration
  
  Methods
  -------

    predict: uses the predict function to estimate the state at step k
      and its uncertainty.  The state is given by:
  
        x = f(x,f_args)

      and the uncertainty is given by
       
        P = F*P*Ft + Q

    update: uses observations z, the observation function, and a
      nonlinear bayesian least squares algorithm to update x and P.
      The Bayesian least squares algorithm solves

        min(||Wd*h(x) - Wd*z||2 + ||Wp*x_new - Wp*x_prior||2)

      where 
   
        Wd.T*Wd = R    and    Wp.T*Wp = P

      and x_new is the new value of x being solved for while x_prior
      is the current value of x. The new value of x is found by
      solving

        | Wd*h(x) |   | Wd*z       |
        |         | = |            |
        | Wp*x    |   | Wp*x_prior |

      for x with a nonlinear least squares algorithm.

  '''    
  def __init__(self,prior,prior_cov,transition,observation):
    self.i = 0
    self.transition = transition
    self.observation = observation
    self.store = [{'prior':prior,
                   'prior_cov':prior_cov,
                   'post':None,
                   'post_cov':None,
                   'smooth':None,
                   'smooth_cov':None,
                   'transition':None,
                   'predicted':None}]

  def predict(self,*args,**kwargs):
    def f(*args,**kwargs):
      return self.transition(*args,**kwargs)[0]

    n = {}
    c = self.store[self.i]
    assert c['post'] is not None

    F = jacobian_fd(c['post'],f,
                    system_args=args,
                    system_kwargs=kwargs)

    Q = self.transition(c['post'],*args,**kwargs)[1]
    c['transition'] = F
    n['prior'] = F.dot(c['post'])
    n['prior_cov'] = F.dot(c['post_cov']).dot(F.transpose()) + Q

    self.store += [n]
    self.i += 1

  def update(self,z,R,*args,**kwargs):
    c = self.store[self.i]
    out = nonlin_lstsq(self.observation,
                       z,c['prior'],
                       system_args=args,
                       system_kwargs=kwargs,
                       data_uncertainty = R,
                       prior_uncertainty = c['prior_cov'],
                       LM_damping=False,
                       output=['solution',
                               'solution_uncertainty',
                               'predicted'])
    c['post'] = out[0]
    c['post_cov'] = out[1]
    c['predicted'] = out[2]

  def smooth(self):
    n = self.i
    clast = self.store[n]
    clast['smooth'] = clast['post']
    clast['smooth_cov'] = clast['post_cov']
    print(clast)
    #self.state_smooth_cov[-1] = self.state_post_cov[-1]
    for n in range(self.i)[::-1]:
      print(n)
      cnext = self.store[n+1]
      ccurr = self.store[n]
      C = ccurr['post_cov'].dot(
          ccurr['transition'].transpose()).dot(
          np.linalg.inv(cnext['prior_cov']))

      ccurr['smooth'] = (ccurr['post'] + 
                         C.dot( 
                         cnext['smooth'] -  
                         cnext['prior']))
      ccurr['smooth_cov'] = (ccurr['post_cov'] + 
                             C.dot(
                             cnext['smooth_cov'] - 
                             cnext['prior_cov']).dot(
                             C.transpose()))

    

                    
    
  





  
