#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ._base import BasePlasticity
from .optimizer import Optimizer, SGD
from .weights import BaseWeights, Normal
from plasticity.lib.hopfield import _Hopfield

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']
__all__ = ['Hopfield']

class Hopfield (BasePlasticity):

  '''
  Hopfield and Krotov implementation of the BCM algorithm [1]_.

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

    weights_init : BaseWeights object (default="Normal")
      Weights initialization strategy.

    epochs_for_convergency : int (default=None)
      Number of stable epochs requested for the convergency.
      If None the training proceeds up to the maximum number of epochs (num_epochs).

    convergency_atol : float (default=0.01)
      Absolute tolerance requested for the convergency

    decay : float (default=0.)
      Weight decay scale factor.

    p : float (default=2.)
      Lebesgue norm of the weights

    k : int (default=2)
      Ranking parameter, must be integer that is bigger or equal than 2

    random_state : int (default=42)
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
  Hopfield(batch_size=100, outputs=100, num_epochs=10, random_state=42, epsilon=0.02, precision=1e-30)
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

  .. image:: ../../../img/Hopfield_weights.gif

  References
  ----------
  .. [1] Dmitry Krotov, and John J. Hopfield. Unsupervised learning by competing hidden units,
         PNAS, 2019, www.pnas.org/cgi/doi/10.1073/pnas.1820458116
  '''

  def __init__(self, outputs : int = 100, num_epochs : int = 100,
      batch_size : int = 100, optimizer : 'Optimizer' = SGD(lr=2e-2),
      delta : float = .4,
      weights_init : 'BaseWeights' = Normal(mu=0., std=1.),
      epochs_for_convergency : int = None,
      convergency_atol : float = 0.01,
      decay : float = 0.,
      p : float = 2., k : int = 2,
      random_state : int = 42, verbose=True):

    super (Hopfield, self).__init__(model=_Hopfield, outputs=outputs, num_epochs=num_epochs,
                                    batch_size=batch_size, activation='Linear',
                                    optimizer=optimizer,
                                    weights_init=weights_init,
                                    epochs_for_convergency=epochs_for_convergency,
                                    convergency_atol=convergency_atol,
                                    random_state=random_state, delta=delta, p=p, k=k,
                                    verbose=verbose)
