#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from plasticity.lib.weights_initialization import _weights_initialization
from plasticity.utils.misc import _check_weights_init

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


class BaseWeights (object):

  '''
  Weights initializer (aka BaseWeights) object

  Parameters
  ----------
    init_type : str or int
      Initialization rule to apply

    mu : float (default=0.0)
      Mean of the normal distribution

    sigma : float (default=1.0)
      Standard deviation of the normal distribution

    scale : float (default=1.0)
      Domain of the uniform distribution

    seed : int (default=42)
      Random seed generator
  '''

  def __init__ (self, init_type : str, mu : float = 0., sigma : float = 1., scale : float = 1., seed : int = 42):

    self.init_type, init_index = _check_weights_init(init_type)
    self.mu = mu
    self.sigma = sigma
    self.scale = scale
    self.seed = seed
    self._object = _weights_initialization(init_index, mu, sigma, scale, seed)

  def __repr__ (self) -> str:
    '''
    Printer of BaseWeights informations
    '''
    class_name = self.__class__.__qualname__

    try:
      params = super(type(self), self).__init__.__code__.co_varnames
    except AttributeError:
      params = self.__init__.__code__.co_varnames

    params = set(params) - {'self', 'init_index', 'init_type'}
    args = ', '.join(['{0}={1}'.format(k, str(getattr(self, k)))
                      if not isinstance(getattr(self, k), str) else '{0}="{1}"'.format(k, str(getattr(self, k)))
                      for k in params])
    return '{0}({1})'.format(class_name, args)

class Zeros (BaseWeights):

  '''
  Initialize weights with zero values
  '''

  def __init__ (self):

    super(Zeros, self).__init__(init_type='Zeros')

class Ones (BaseWeights):

  '''
  Initialize weights with one values
  '''

  def __init__ (self):

    super(Ones, self).__init__(init_type='Ones')

class Uniform (BaseWeights):

  '''
  Sample initial weights from the uniform distribution.

  Parameters are sampled from U(a, b).

  Parameters
  ----------
    scale : float or tuple.
      When std is None then range determines a, b. If range is a float the
      weights are sampled from U(-range, range). If range is a tuple the
      weights are sampled from U(range[0], range[1]).
  '''

  def __init__ (self, scale):

    super(Uniform, self).__init__(init_type='Uniform', scale=scale)

class Normal (BaseWeights):

  '''
  Sample initial weights from the Gaussian distribution.

  Initial weight parameters are sampled from N(mean, std).

  Parameters
  ----------
    mu : float (default=0.)
      Mean of initial parameters.

    std : float (default=1.)
      Std of initial parameters.
  '''

  def __init__ (self, mu=0., std=1.):

    super(Normal, self).__init__(init_type='Normal', mu=mu, sigma=std)

class LecunUniform (BaseWeights):

  '''
  LeCun uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(3 / inputs)` [2]_
  where `inputs` is the number of input units in the weight matrix.

  References
  ----------
  .. [2] LeCun 98, Efficient Backprop, http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  '''

  def __init__ (self):

    super(LecunUniform, self).__init__(init_type='LecunUniform')

class GlorotUniform (BaseWeights):

  '''
  Glorot uniform initializer, also called Xavier uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / (inputs + outputs))` [3]_
  where `inputs` is the number of input units in the weight matrix
  and `outputs` is the number of output units in the weight matrix.

  References
  ----------
  .. [3] Glorot & Bengio, AISTATS 2010. http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
  '''

  def __init__ (self):

    super(GlorotUniform, self).__init__(init_type='GlorotUniform')

class LecunNormal (BaseWeights):

  '''
  Lecun normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(1 / inputs))` [4]_
  where `inputs` is the number of input units in the weight matrix.

  References
  ----------
  .. [4] LeCun 98, Efficient Backprop, http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  '''

  def __init__ (self):
    super(LecunNormal, self).__init__(init_type='LecunNormal')

class GlorotNormal (BaseWeights):

  '''
  Glorot normal initializer, also called Xavier normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(2 / (inputs + outputs))` [5]_
  where `inputs` is the number of input units in the weight matrix
  and `outputs` is the number of output units in the weight matrix.

  References
  ----------
  .. [5] Glorot & Bengio, AISTATS 2010. http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
  '''

  def __init__ (self):

    super(GlorotNormal, self).__init__(init_type='GlorotNormal')

class HeUniform (BaseWeights):

  '''
  He uniform variance scaling initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / inputs)` [6]_
  where `inputs` is the number of input units in the weight matrix.

  References
  ----------
  .. [6] He et al., http://arxiv.org/abs/1502.01852
  '''

  def __init__ (self):

    super(HeUniform, self).__init__(init_type='HeUniform')

class HeNormal (BaseWeights):

  '''
  He normal initializer.

  It draws samples from a truncated normal distribution centered on 0
  with `stddev = sqrt(2 / inputs)` [7]_
  where `inputs` is the number of input units in the weight matrix.

  References
  ----------
  .. [7] He et al., http://arxiv.org/abs/1502.01852
  '''

  def __init__ (self):

    super(HeNormal, self).__init__(init_type='HeNormal')
