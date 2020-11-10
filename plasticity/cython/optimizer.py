#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from plasticity.lib.update_args import _update_args
from plasticity.utils.misc import _check_update

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


class Optimizer (object):

  '''
  Update arguments (aka Optimizer) object

  Parameters
  ----------
    update_type : str or int
      Update rule to apply

    learning_rate : float (default=2e-2)
      Learning rate value

    momentum : float (default=0.9)
      Momentum parameter

    decay : float (default=1e-4)
      Decay parameter

    B1 : float (default=0.9)
      Adam-like parameter

    B2 : float (default=0.999)
      Adam-like parameter

    rho : float (default=0.0)
      Decay factor in RMSProp and AdaDelta

    l2norm : bool (default=False)
      Normalize the gradient values according to their l2 norms

    clip_value : bool (default=False)
      Clip gradient values between -1 and 1
  '''

  def __init__ (self, update_type, learning_rate=1e-3, momentum=.9, decay=1e-4, B1=.9, B2=.999, rho=0., l2norm=False, clip_value=False):

    self.update_type, update_index = _check_update(update_type)
    self.l2norm = l2norm
    self.clip_value = clip_value

    self._object = _update_args(update_index, learning_rate, momentum, decay, B1, B2, rho, l2norm, clip_value)

  @property
  def learning_rate (self):
    '''
    Return the current learning rate parameter
    '''
    return self._object.get_learning_rate

  @property
  def momentum (self):
    '''
    Return the current momentum parameter
    '''
    return self._object.get_momentum

  @property
  def decay (self):
    '''
    Return the current decay parameter
    '''
    return self._object.get_decay

  @property
  def B1 (self):
    '''
    Return the current B1 parameter
    '''
    return self._object.get_B1

  @property
  def B2 (self):
    '''
    Return the current B2 parameter
    '''
    return self._object.get_B2

  @property
  def rho (self):
    '''
    Return the current rho parameter
    '''
    return self._object.get_rho

  @property
  def is_norm (self):
    '''
    Return True if the gradient normalization is enabled
    '''
    return bool(self._object.get_l2norm)

  @property
  def is_clip (self):
    '''
    Return True if the gradient clipping is enabled
    '''
    return bool(self._object.get_clip)

  def __repr__ (self):
    '''
    Printer of Optimizer informations
    '''
    class_name = self.__class__.__qualname__

    try:
      params = super(type(self), self).__init__.__code__.co_varnames
    except AttributeError:
      params = self.__init__.__code__.co_varnames

    params = set(params) - {'self', 'update_index'}
    args = ', '.join(['{0}={1}'.format(k, str(getattr(self, k)))
                      if not isinstance(getattr(self, k), str) else '{0}="{1}"'.format(k, str(getattr(self, k)))
                      for k in params])
    return '{0}({1})'.format(class_name, args)

class SGD (Optimizer):

  '''
  Stochastic Gradient Descent specialization

  Update the parameters according to the rule

    parameter -= learning_rate * gradient

  Parameters
  ----------
    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, *args, **kwargs):

    super(SGD, self).__init__(update_type='SGD', *args, **kwargs)

class RMSProp (Optimizer):

  '''
  RMSProp specialization

  Parameters
  ----------
    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, *args, **kwargs):

    super(RMSProp, self).__init__(update_type='RMSprop', *args, **kwargs)

class Momentum (Optimizer):

  '''
  Stochastic Gradient Descent with Momentum specialiation

  Parameters
  ----------
    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, *args, **kwargs):

    super(Momentum, self).__init__(update_type='Momentum', *args, **kwargs)

class NesterovMomentum (Optimizer):

  '''
  Stochastic Gradient Descent with Nesterov Momentum specialiation.

  Parameters
  ----------
    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, *args, **kwargs):

    super(NesterovMomentum, self).__init__(update_type='NesterovMomentum', *args, **kwargs)

class Adam (Optimizer):

  '''
  Adam specialization.

  Parameters
  ----------
    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, *args, **kwargs):

    super(Adam, self).__init__(update_type='Adam', *args, **kwargs)

class Adagrad (Optimizer):

  '''
  Adagrad optimizer specialization.

  Parameters
  ----------
    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, *args, **kwargs):

    super(Adagrad, self).__init__(update_type='Adagrad', *args, **kwargs)

class Adadelta (Optimizer):

  '''
  Adadelta specialization.

  Parameters
  ----------
    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, *args, **kwargs):

    super(Adadelta, self).__init__(update_type='Adadelta', *args, **kwargs)

class Adamax (Optimizer):

  '''
  Adamax specialization

  Parameters
  ----------
    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, *args, **kwargs):

    super(Adamax, self).__init__(update_type='Adamax', *args, **kwargs)
