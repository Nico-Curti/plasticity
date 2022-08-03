#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ._base import BasePlasticity
from .optimizer import Optimizer, SGD
from .weights import BaseWeights, Normal
from plasticity.lib.bcm import _BCM

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']
__all__ = ['BCM']

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
      Number of epochs for model convergence

    batch_size : int (default=10)
      Size of the minibatch

    activation : string or Activations object (default='Logistic')
      Activation function to apply

    optimizer : Optimizer (default=SGD)
      Optimizer object (derived by the base class Optimizer)

    weights_init : BaseWeights object (default="Normal")
      Weights initialization strategy.

    epochs_for_convergence : int (default=None)
      Number of stable epochs requested for the convergence.
      If None the training proceeds up to the maximum number of epochs (num_epochs).

    convergence_atol : float (default=0.01)
      Absolute tolerance requested for the convergence

    decay : float (default=0.)
      Weight decay scale factor.

    memory_factor : float (default=0.5)
      Memory factor for weighting the theta updates.

    interaction_strength : float (default=0.)
      Set the lateral interaction strength between weights

    random_state : int (default=42)
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
  .. [1] Squadrani L., Curti N., Giampieri E., Remondini D., Blais B., Castellani G.
         Effectiveness of Biologically Inspired Neural Network Models in Learning and Patterns
         Memorization. Entropy 2022, 24, 682. https://doi.org/10.3390/e24050682
  .. [2] Castellani G., Intrator N., Shouval H.Z., Cooper L.N. Solutions of the BCM learning rule
         in a network of lateral interacting nonlinear neurons, Network Computation in Neural Systems,
         10.1088/0954-898X/10/2/001
  '''

  def __init__(self, outputs : int = 100, num_epochs : int = 100,
      batch_size : int = 100, activation : str = 'Logistic',
      optimizer : 'Optimizer' = SGD(lr=2e-2),
      weights_init : 'BaseWeights' = Normal(mu=0., std=1.),
      epochs_for_convergence : int = None,
      convergence_atol : float = 0.01,
      decay : float = 0.,
      memory_factor: float = 0.5,
      interaction_strength : float = 0., random_state : int = 42,
      verbose : bool = True):

    super (BCM, self).__init__(model=_BCM, outputs=outputs, num_epochs=num_epochs,
                               batch_size=batch_size, activation=activation,
                               optimizer=optimizer,
                               weights_init=weights_init,
                               epochs_for_convergence=epochs_for_convergence,
                               convergence_atol=convergence_atol,
                               decay=decay,
                               memory_factor=memory_factor,
                               random_state=random_state, verbose=verbose,
                               interaction_strength=interaction_strength)

