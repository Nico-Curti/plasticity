#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from plasticity.lib.update_args import _update_args
from plasticity.utils.misc import _check_update

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


class Optimizer:

  def __init__ (self, update_type, learning_rate=1e-3, momentum=.9, decay=1e-4, B1=.9, B2=.999, rho=0., l2norm=False, clip_value=False):
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
      TODO

    l2norm : bool (default=False)
      Normalize the gradient values according to their l2 norms

    clip_value : bool (default=False)
      Clip gradient values between -1 and 1
    '''

    self.update_type, update_index = _check_update(update_type)
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
  def iteration (self):
    '''
    Return the current iteration parameter
    '''
    return self._object.get_iteration

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

  def __str__ (self):
    '''
    Printer of Optimizer informations
    '''
    return '{0} (iteration={1:d}, learning_rate={2:.3f}, momentum={3:.3f}, decay={4:.3f}, B1={5:.3f}, B2={6:.3f}, rho={7:.3f}, l2norm={8}, clip_value={9})'.format(
            self.update_type, self.iteration, self.learning_rate, self.momentum, self.decay, self.B1, self.B2, self.rho, self.is_norm, self.is_clip
            )

class SGD (Optimizer):

  def __init__ (self, *args, **kwargs):

    super(SGD, self).__init__(update_type='SGD', *args, **kwargs)

class RMSProp (Optimizer):

  def __init__ (self, *args, **kwargs):

    super(RMSProp, self).__init__(update_type='RMSprop', *args, **kwargs)

class Momentum (Optimizer):

  def __init__ (self, *args, **kwargs):

    super(Momentum, self).__init__(update_type='Momentum', *args, **kwargs)

class NesterovMomentum (Optimizer):

  def __init__ (self, *args, **kwargs):

    super(NesterovMomentum, self).__init__(update_type='NesterovMomentum', *args, **kwargs)

class Adam (Optimizer):

  def __init__ (self, *args, **kwargs):

    super(Adam, self).__init__(update_type='Adam', *args, **kwargs)

class Adagrad (Optimizer):

  def __init__ (self, *args, **kwargs):

    super(Adagrad, self).__init__(update_type='Adagrad', *args, **kwargs)

class Adadelta (Optimizer):

  def __init__ (self, *args, **kwargs):

    super(Adadelta, self).__init__(update_type='Adadelta', *args, **kwargs)

class Adamax (Optimizer):

  def __init__ (self, *args, **kwargs):

    super(Adamax, self).__init__(update_type='Adamax', *args, **kwargs)
