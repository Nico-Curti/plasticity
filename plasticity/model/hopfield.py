#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from plasticity.model._base import BasePlasticity
from .optimizer import SGD

__author__  = ['Nico Curti', 'SimoneGasperini']
__email__ = ['nico.curit2@unibo.it', 'simone.gasperini2@studio.unibo.it']


class Hopfield (BasePlasticity):

  '''
  Hopfield and Krotov implementation of the BCM algorithm.

  Parameters
  ----------
    outputs : int (default=100)
      Number of hidden units

    num_epochs : int (default=100)
      Number of epochs for model convergency

    batch_size : int (default=10)
      Size of the minibatch

    optimizer : Optimizer (default=SGD)
      Optimizer object (derived by the base class Optimizer)

    delta : float (default=0.4)
      Strength of the anti-hebbian learning

    mu : float (default=0.)
      Mean of the gaussian distribution that initializes the weights

    sigma : float (default=1.)
      Standard deviation of the gaussian distribution that initializes the weights

    p : float (default=2.)
      Lebesgue norm of the weights

    k : int (default=2)
      Ranking parameter, must be integer that is bigger or equal than 2

    precision : float (default=1e-30)
      Parameter that controls numerical precision of the weight updates

    seed : int (default=42)
      Random seed for weights generation

    verbose : bool (default=True)
      Turn on/off the verbosity

  Examples
  --------
  >>> from sklearn.datasets import fetch_openml
  >>> import pylab as plt
  >>> from plasticity.model import Hopfield
  >>>
  >>> X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)
  >>> X *= 1. / 255
  >>> model = Hopfield(outputs=100, num_epochs=10)
  >>> model.fit(X)
  Hopfield(batch_size=100, outputs=100, sigma=1.0, mu=0.0,
  num_epochs=10, seed=42, precision=1e-30)
  >>>
  >>> # view the memorized weights
  >>> w = model.weights[0].reshape(28, 28)
  >>> nc = np.max(np.abs(w))
  >>>
  >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
  >>> im = ax.imshow(w, cmap='bwr', vmin=-nc, vmax=nc)
  >>> fig.colorbar(im, ticks=[np.min(w), 0, np.max(w)])
  >>> ax.axis("off")
  >>> plt.show()

  .. image:: ../../../img/Hopfield_weights.png

  References
  ----------
  - Dmitry Krotov, and John J. Hopfield. Unsupervised learning by competing hidden units,
    PNAS, 2019, www.pnas.org/cgi/doi/10.1073/pnas.1820458116
  '''

  def __init__(self, outputs=100, num_epochs=100,
      batch_size=100, delta=.4,
      optimizer=SGD(learning_rate=2e-2),
      mu=0., sigma=1.,
      p=2., k=2,
      precision=1e-30,
      seed=42, verbose=True):

    self.delta = delta
    self.p = p
    self.k = k

    super (Hopfield, self).__init__(outputs=outputs, num_epochs=num_epochs,
                               batch_size=batch_size, activation='Linear',
                               optimizer=optimizer,
                               mu=mu, sigma=sigma,
                               precision=precision,
                               seed=seed, verbose=verbose)

  def _weights_update (self, X, output):
    '''
    Approximation introduced by Krotov.
    Instead of solving dynamical equations we use the currents as a proxy
    for ranking of the final activities as suggested in [0].

    Notes
    -----
    .. note::
      This is the core function of the Hopfield class since it implements
      the Hopfield learning rule.
    '''

    order = np.argsort(output, axis=0)
    yl = np.zeros_like(output, dtype=float)
    yl[order[-1, :], range(self.batch_size)] = 1.
    yl[order[-self.k, :], range(self.batch_size)] = - self.delta

    xx = np.sum(yl * output, axis=1, keepdims=True)
    #ds = yl @ X - xx * self.weights # TODO: convert to einsum
    ds = np.einsum('ij, jk -> ik', yl, X, optimize=True) - xx * self.weights

    nc = np.max(np.abs(ds))
    nc = 1. / max(nc, self.precision)

    return ds * nc


  def _fit (self, X):
    '''
    Core function for the fit member
    '''

    return super(Hopfield, self)._fit(X=X, norm=True)


if __name__ == '__main__':

  import pylab as plt
  from sklearn.datasets import fetch_openml

  # Download the MNIST dataset
  X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

  # normalize the sample into [0, 1]
  X *= 1. / 255

  model = Hopfield(outputs=100, num_epochs=10)
  model.fit(X)

  weights = model.weights.reshape(-1, 28, 28)
  w = weights[0]
  nc = np.max(np.abs(w))

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
  im = ax.imshow(w, cmap='bwr', vmin=-nc, vmax=nc)
  fig.colorbar(im, ticks=[np.min(w), 0, np.max(w)])
  ax.axis('off')
  plt.show()
