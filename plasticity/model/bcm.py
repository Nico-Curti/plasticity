#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from plasticity.model._base import BasePlasticity
from plasticity.model.optimizer import SGD
from plasticity.model.weights import Normal

__author__  = ['Nico Curti', 'SimoneGasperini']
__email__ = ['nico.curit2@unibo.it', 'simone.gasperini2@studio.unibo.it']


class BCM (BasePlasticity):

  '''
  Bienenstock, Cooper and Munro algorithm (BCM) [1]_.

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
      Maximum number of epochs for model convergency

    batch_size : int (default=10)
      Size of the minibatch

    activation : string or Activations object (default='Logistic')
      Activation function to apply

    optimizer : Optimizer (default=SGD)
      Optimizer object (derived by the base class Optimizer)

    weights_init : BaseWeights object (default="Normal")
      Weights initialization strategy.

    interaction_strength : float (default=0.)
      Set the lateral interaction strenght between weights

    precision : float (default=1e-30)
      Parameter that controls numerical precision of the weight updates

    epochs_for_convergency : int (default=None)
      Number of stable epochs requested for the convergency.
      If None the training proceeds up to the maximum number of epochs (num_epochs).

    convergency_atol : float (default=0.01)
      Absolute tolerance requested for the convergency

    random_state : int (default=None)
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
  BCM(batch_size=100, outputs=100, num_epochs=10, random_state=42, epsilon=0.02, precision=1e-30)
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

  .. image:: ../../../img/BCM_weights.gif

  References
  ----------
  .. [1] Castellani G., Intrator N., Shouval H.Z., Cooper L.N. Solutions of the BCM learning rule
         in a network of lateral interacting nonlinear neurons, Network Computation in Neural Systems, 10.1088/0954-898X/10/2/001
  '''

  def __init__(self, outputs=100, num_epochs=100,
      batch_size=100, activation='Logistic',
      optimizer=SGD(learning_rate=2e-2),
      weights_init=Normal(mu=0., std=1.),
      interaction_strength=0.,
      precision=1e-30,
      epochs_for_convergency=None,
      convergency_atol=0.01,
      random_state=None, verbose=True):

    self._interaction_matrix = self._weights_interaction(interaction_strength, outputs)
    self.interaction_strength = interaction_strength

    super (BCM, self).__init__(outputs=outputs, num_epochs=num_epochs,
                               batch_size=batch_size, activation=activation,
                               optimizer=optimizer,
                               weights_init=weights_init,
                               precision=precision,
                               epochs_for_convergency=epochs_for_convergency,
                               convergency_atol=convergency_atol,
                               random_state=random_state, verbose=verbose)

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

    Parameters
    ----------
      X : array-like (2D)
        Input array of data

      output : array-like (2D)
        Output of the model estimated by the predict function

    Returns
    -------
      weight_update : array-like (2D)
        Weight updates matrix to apply

      theta : array-like (1D)
        Array of learning progress

    Notes
    -----
    .. note::
      This is the core function of the BCM class since it implements
      the BCM learning rule.
    '''

    theta = np.mean(output**2, axis=1, keepdims=True)
    phi = output * (output - theta) * (1. / (theta + self.precision))

    #dw = phi @ X
    dw = np.einsum('ij, jk -> ik', phi, X, optimize=True)

    nc = np.max(np.abs(dw))
    nc = 1. / max(nc, self.precision)

    return dw * nc, theta


  def _fit (self, X):
    '''
    Core function for the fit member
    '''

    return super(BCM, self)._fit(X=X, norm=False)


  def _predict (self, X):
    '''
    Core function for the predict member
    '''

    # return self.activation.activation( self._interaction_matrix @ self.weights @ X.T, copy=True)
    return self.activation.activate(np.einsum('ij, jk, lk -> il', self._interaction_matrix, self.weights, X, optimize=True), copy=True)


if __name__ == '__main__':

  from sklearn.datasets import fetch_openml
  from plasticity.utils import view_weights

  # Download the MNIST dataset
  X, y = fetch_openml(name='mnist_784', version=1, data_id=None, return_X_y=True)
  #X = np.where(X != 0, 1, 0).astype('float')
  # normalize the sample into [0, 1]
  X *= 1. / 255

  model = BCM(outputs=100, num_epochs=10, batch_size=100, activation='logistic',
              interaction_strength=-0.05, optimizer=SGD(lr=2e-2))
  model.fit(X)

  view_weights (model.weights, dims=(28, 28))
