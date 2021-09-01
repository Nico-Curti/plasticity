#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']


class BaseWeights (object):

  '''
  Base class for weights initialization

  References
  ----------
  - https://raw.githubusercontent.com/oujago/NumpyDL/master/npdl/initializations.py
  '''

  def get (self, size : tuple) -> np.ndarray:
    '''
    Initialize the weigths matrix according to the
    specialization

    Parameters
    ----------
      size : tuple
        Weights matrix shape

    Returns
    -------
      weights : array-like
        Matrix of weights with the given shape
    '''
    raise NotImplementedError

  @property
  def name (self) -> str:
    '''
    Get the name of the weight initializer function
    '''
    class_name = self.__class__.__qualname__
    return class_name

  def __repr__ (self) -> str:
    '''
    Printer
    '''
    class_name = self.name
    try:
      params = super(type(self), self).__init__.__code__.co_varnames
    except AttributeError:
      params = self.__init__.__code__.co_varnames

    params = set(params) - {'self', 'args', 'kwargs'}
    args = ', '.join(['{0}={1}'.format(k, str(getattr(self, k)))
                      if not isinstance(getattr(self, k), str) else '{0}="{1}"'.format(k, str(getattr(self, k)))
                      for k in params])
    return '{0}({1})'.format(class_name, args)


class Zeros (BaseWeights):

  '''
  Initialize weights with zero values
  '''

  def __init__ (self):
    super(Zeros, self).__init__()

  def get (self, size : tuple) -> np.ndarray:
    return np.zeros(shape=size, dtype=float)

class Ones (BaseWeights):

  '''
  Initialize weights with one values
  '''

  def __init__ (self):
    super(Ones, self).__init__()

  def get (self, size : tuple) -> np.ndarray:
    return np.ones(shape=size, dtype=float)

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

  def __init__ (self, scale : float = .05):
    self.scale = scale
    super(Uniform, self).__init__()

  def get (self, size : tuple) -> np.ndarray:
    return np.random.uniform(low=-self.scale, high=self.scale, size=size)

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

  def __init__ (self, mu : float = 0., std : float = 1.):
    self.mu = mu
    self.std = std
    super(Normal, self).__init__()

  def get (self, size : tuple) -> np.ndarray:
    return np.random.normal(loc=self.mu, scale=self.std, size=size)

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
    super(LecunUniform, self).__init__()

  def get (self, size : tuple) -> np.ndarray:
    inputs = size[0]
    scale = np.sqrt(3. / inputs)
    return np.random.uniform(low=-scale, high=scale, size=size)

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
    super(GlorotUniform, self).__init__()

  def get (self, size : tuple) -> np.ndarray:
    inputs, outputs = size
    scale = np.sqrt(6. / (inputs + outputs))
    return np.random.uniform(low=-scale, high=scale, size=size)

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
    super(LecunNormal, self).__init__()

  def get (self, size : tuple) -> np.ndarray:
    inputs, outputs = size
    std = np.sqrt(1. / inputs)
    return np.random.normal(loc=0., scale=std, size=size)

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
    super(GlorotNormal, self).__init__()

  def get (self, size : tuple) -> np.ndarray:
    inputs, outputs = size
    std = np.sqrt(2. / (inputs + outputs))
    return np.random.normal(loc=0., scale=std, size=size)

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
    super(HeUniform, self).__init__()

  def get (self, size : tuple) -> np.ndarray:
    inputs = size[0]
    scale = np.sqrt(6. / inputs)
    return np.random.uniform(low=-scale, high=scale, size=size)

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
    super(HeNormal, self).__init__()

  def get (self, size : tuple) -> np.ndarray:
    inputs = size[0]
    std = np.sqrt(2. / inputs)
    return np.random.normal(loc=0., scale=std, size=size)

class Orthogonal (BaseWeights):

  '''
  Intialize weights as Orthogonal matrix.

  Orthogonal matrix initialization [8]_. For n-dimensional shapes where
  n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
  corresponds to the fan-in, so this makes the initialization usable for
  both dense and convolutional layers.

  Parameters
  ----------
    gain : float or 'relu'.
      Scaling factor for the weights. Set this to ``1.0`` for linear and
      sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
      to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
      leakiness ``alpha``. Other transfer functions may need different
      factors.

  References
  ----------
  .. [8] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
         "Exact solutions to the nonlinear dynamics of learning in deep
         linear neural networks." arXiv preprint arXiv:1312.6120 (2013).
  '''

  def __init__ (self, gain : float = 1.):

    self.gain = np.sqrt(2) if gain == 'relu' else gain
    super(Orthogonal, self).__init__()

  def get (self, size : tuple) -> np.ndarray:

    flat_shape = (size[0], np.prod(size[1:]))
    a = np.random.normal(loc=0., scale=1., size=flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(size)
    q = self.gain * q
    return q

class TruncatedNormal (BaseWeights):

  '''
  Generate draws from a truncated normal distribution via rejection sampling.

  Parameters
  ----------
    mean : float or array_like of floats
      The mean/center of the distribution
    std : float or array_like of floats
      Standard deviation (spread or "width") of the distribution.
    out_shape : int or tuple of ints
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.

  Notes
  -----
  The rejection sampling regimen draws samples from a normal distribution
  with mean `mean` and standard deviation `std`, and resamples any values
  more than two standard deviations from `mean`.

  References
  ----------
  - https://raw.githubusercontent.com/ddbourgin/numpy-ml/master/numpy_ml/neural_nets/utils/utils.py
  '''

  def __init__ (self, mu : float = 0., std : float = 1.):
    self.mu = mu
    self.std = std
    super(TruncatedNormal, self).__init__()

  def get (self, size : tuple) -> np.ndarray:
    samples = np.random.normal(loc=self.mu, scale=self.std, size=size)
    reject = np.logical_or(samples >= self.mu + 2 * self.std, samples <= self.mu - 2 * self.std)

    while any(reject.flatten()):
      resamples = np.random.normal(loc=self.mu, scale=self.std, size=reject.sum())
      samples[reject] = resamples
      reject = np.logical_or(samples >= self.mu + 2 * self.std, samples <= self.mu - 2 * self.std)

    return samples

