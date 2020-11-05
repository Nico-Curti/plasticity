#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ._base import BasePlasticity
from plasticity.lib.bcm import _BCM

__author__  = ['Nico Curti']
__email__ = ['nico.curit2@unibo.it']


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
    outputs : int, default=100
      Number of hidden units

    num_epochs : int, default=100
      Number of epochs for model convergency

    batch_size : int, default=10
      Size of the minibatch

    activation : string or Activations object (default = 'Logistic')
      Activation function to apply

    mu : float, default=0.
      Mean of the gaussian distribution that initializes the weights

    sigma : float, default=1.
      Standard deviation of the gaussian distribution that initializes the weights

    interaction_strenght : float, default=0.
      Set the lateral interaction strenght between weights

    epsilon : float, default=2e-2
      Learning rate

    seed : int, default=42
      Random seed for weights generation

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

  References
  ----------

  [0] Castellani G., Intrator N., Shouval H.Z., Cooper L.N. Solutions of the BCM learning
  rule in a network of lateral interacting nonlinear neurons, Network Computation in Neural Systems,
  10.1088/0954-898X/10/2/001

  '''

  def __init__(self, outputs=100, num_epochs=100,
      batch_size=100, activation='Logistic',
      mu=0., sigma=1., epsilon=2e-2,
      interaction_strenght=0., seed=42):

    super (BCM, self).__init__(model=_BCM, outputs=outputs, num_epochs=num_epochs,
                               batch_size=batch_size, activation=activation,
                               mu=mu, sigma=sigma,
                               epsilon=epsilon, seed=seed,
                               interaction_strenght=interaction_strenght)

