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

    lr : float (default=1e-3)
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
  '''

  def __init__ (self, update_type : str,
      lr : float = 1e-3, momentum : float = .9, decay : float = 1e-4,
      B1 : float = .9, B2 : float = .999, rho : float = 0.):

    self.update_type, update_index = _check_update(update_type)

    self._object = _update_args(update_index, lr, momentum, decay, B1, B2, rho)

  @property
  def learning_rate (self) -> float:
    '''
    Return the current learning rate parameter
    '''
    return self._object.get_learning_rate

  @property
  def momentum (self) -> float:
    '''
    Return the current momentum parameter
    '''
    return self._object.get_momentum

  @property
  def decay (self) -> float:
    '''
    Return the current decay parameter
    '''
    return self._object.get_decay

  @property
  def B1 (self) -> float:
    '''
    Return the current B1 parameter
    '''
    return self._object.get_B1

  @property
  def B2 (self) -> float:
    '''
    Return the current B2 parameter
    '''
    return self._object.get_B2

  @property
  def rho (self) -> float:
    '''
    Return the current rho parameter
    '''
    return self._object.get_rho

  def __repr__ (self) -> str:
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
