#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from plasticity.model._base import BasePlasticity
from .optimizer import SGD

__author__  = ['Nico Curti', 'SimoneGasperini']
__email__ = ['nico.curit2@unibo.it', 'simone.gasperini2@studio.unibo.it']


class BCM (BasePlasticity):

  '''
  Bienenstock, Cooper and Munro algorithm (BCM).

  The idea of BCM theory is that for a random sequence of input patterns a synapse
  is learning to differentiate between those stimuli that excite the postsynaptic
  neuron strongly and those stimuli that excite that neuron weakly.
  Learned BCM feature detectors cannot, however, be simply used as the lowest layer
  of a feedforward network so that the entire network is competitive to a network of
  the same size trained with backpropagation algorithm end-to-end.

  Parameters
  ----------
    outputs : int (default=100)
      Number of hidden units

    num_epochs : int (default=100)
      Number of epochs for model convergency

    batch_size : int (default=10)
      Size of the minibatch

    activation : string or Activations object (default='Logistic')
      Activation function to apply

    optimizer : Optimizer (default=SGD)
      Optimizer object (derived by the base class Optimizer)

    mu : float (default=0.)
      Mean of the gaussian distribution that initializes the weights

    sigma : float (default=1.)
      Standard deviation of the gaussian distribution that initializes the weights

    interaction_strength : float (default=0.)
      Set the lateral interaction strenght between weights

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
  >>> from plasticity.model import BCM
  >>>
  >>> X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)
  >>> X *= 1. / 255
  >>> model = BCM(outputs=100, num_epochs=10)
  >>> model.fit(X)
  BCM(batch_size=100, outputs=100, sigma=1.0, mu=0.0,
  num_epochs=10, seed=42, epsilon=0.02, precision=1e-30)
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

  .. image:: ../../../img/BCM_weights.png

  References
  ----------
  - Castellani G., Intrator N., Shouval H.Z., Cooper L.N. Solutions of the BCM learning rule
    in a network of lateral interacting nonlinear neurons, Network Computation in Neural Systems, 10.1088/0954-898X/10/2/001
  '''

  def __init__(self, outputs=100, num_epochs=100,
      batch_size=100, activation='Logistic',
      optimizer=SGD(learning_rate=2e-2),
      mu=0., sigma=1., interaction_strength=0.,
      precision=1e-30,
      seed=42, verbose=True):

    self._interaction_matrix = self._weights_interaction(interaction_strength, outputs)
    self.interaction_strength = interaction_strength

    super (BCM, self).__init__(outputs=outputs, num_epochs=num_epochs,
                               batch_size=batch_size, activation=activation,
                               optimizer=optimizer,
                               mu=mu, sigma=sigma,
                               precision=precision,
                               seed=seed, verbose=verbose)

  def _weights_interaction (self, strength, outputs):
    '''
    Set the interaction matrix between weights' connections

    Parameters
    ----------
      strength : float
        Interaction strength between weights

      outputs : int
        Number of hidden units

    Returns
    -------
      interaction_matrix : array-like
        Matrix of interactions between weights
    '''

    if strength != 0.:
      L = np.full(fill_value=-strength, shape=(outputs, outputs))
      L[np.eye(*L.shape, dtype=bool)] = 1

      return np.linalg.inv(L)

    else:
      return np.eye(M=outputs, N=outputs)

  def _weights_update (self, X, output):
    '''
    Compute the weights update using the BCM learning rule.

    Notes
    -----
    .. note::
      This is the core function of the BCM class since it implements
      the BCM learning rule.
    '''

    theta = np.mean(output**2, axis=1, keepdims=True)
    phi = output * (output - theta)
    output = self.activation.activate(output)

    #dw = self._interaction_matrix @ phi * self.gradient(output) @ X
    dw = np.einsum('ij, jk, ik, kl -> il', self._interaction_matrix, phi, self.activation.gradient(output), X, optimize=True)

    nc = np.max(np.abs(dw))
    nc = 1. / max(nc, self.precision)

    return dw * nc

  def _fit (self, X, view=False):
    '''
    Core function for the fit member
    '''

    return super(BCM, self)._fit(X=X, norm=False, view=view)


if __name__ == '__main__':

  import pylab as plt
  from sklearn.datasets import fetch_openml

  # Download the MNIST dataset
  X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)

  # normalize the sample into [0, 1]
  X *= 1. / 255

  model = BCM(outputs=100, num_epochs=10, activation='Logistic', epsilon=.04)
  model.fit(X)

  w = model.weights[0].reshape(28, 28)
  nc = np.max(np.abs(w))

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
  im = ax.imshow(w, cmap='bwr', vmin=-nc, vmax=nc)
  fig.colorbar(im, ticks=[np.min(w), 0, np.max(w)])
  ax.axis('off')
  plt.show()
