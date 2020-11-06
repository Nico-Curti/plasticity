#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

from plasticity.utils import _check_activation

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

__author__  = ['Nico Curti', 'SimoneGasperini']
__email__ = ['nico.curit2@unibo.it', 'simone.gasperini2@studio.unibo.it']


class BasePlasticity (BaseEstimator, TransformerMixin):

  '''
  Abstract base class for plasticity models

  Parameters
  ----------
    outputs : int, default=100
      Number of hidden units

    num_epochs : int, default=100
      Number of epochs for model convergency

    batch_size : int, default=100
      Size of the minibatch

    activation : str, default="linear"
      Name of the activation function

    mu : float, default=0.
      Mean of the gaussian distribution that initializes the weights

    sigma : float, default=1.
      Standard deviation of the gaussian distribution that initializes the weights

    epsilon : float, default=2e-2
      Starting learning rate

    precision : float, default=1e-30
      Parameter that controls numerical precision of the weight updates

    seed : int, default=42
      Random seed for weights generation
  '''

  def __init__ (self, outputs=100, num_epochs=100,
      activation='linear', batch_size=100, mu=0., sigma=1.,
      epsilon=2e-2, precision=1e-30, seed=42):

    _, activation = _check_activation(self, activation)

    self.outputs = outputs
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.activation = activation.activate
    self.gradient = activation.gradient
    self.mu = mu
    self.sigma = sigma
    self.epsilon = epsilon
    self.precision = precision
    self.seed = seed

  def _weights_update(self, X, output):
    '''
    Compute the weights update using the given learning rule.
    '''
    raise NotImplementedError

  def _lebesque_norm(self):
    '''
    Apply the Lebesgue norm to the weights.
    '''
    if self.p != 2:
      sign = np.sign(self.weights)
      self.weights = sign * np.absolute(self.weights)**(self.p - 1)

  def _fit_step (self, X, norm, epsilon):
    '''
    Core function of fit step (forward + backward + updates).
    We divide the step into a function to allow an easier visualization
    of the weight matrix (if necessary).
    '''

    if norm:
      self._lebesque_norm()

    # predict the encoded values
    output = self._predict(X)

    # update weights
    w_update = self._weights_update(X, output)

    self.weights[:] += epsilon * w_update


  def _fit (self, X, norm=False, view=False):
    '''
    Core function for the fit member
    '''

    epsilon = np.linspace(start=self.epsilon, stop=self.epsilon*(1. - (self.num_epochs - 1) / self.num_epochs), num=self.num_epochs)

    num_samples, _ = X.shape
    indices = np.arange(0, num_samples).astype('int64')
    num_batches = num_samples // self.batch_size

    for epoch, epsil in enumerate(epsilon):
      print('Epoch {:d}/{:d}'.format(epoch + 1, self.num_epochs))

      # random shuffle the input
      np.random.shuffle(X)

      batches = np.lib.stride_tricks.as_strided(indices, shape=(num_batches, self.batch_size), strides=(self.batch_size * 8, 8))

      for batch in tqdm(batches):

        batch_data = X[batch, ...]

        self._fit_step(X=batch_data, norm=norm, epsilon=epsil)


    return self

  def fit (self, X, y=None, view=False):
    '''
    Fit the Plasticity model weights.

    Parameters
    ----------
      X : array-like of shape (n_samples, n_features)
        The training input samples

      y : array-like, default=None
        The array of labels

      view : bool
        Switch if plot the weight matrix at each iteration or not

    Returns
    -------
      self : object
        Return self

    Notes
    -----
    .. note::
      The model tries to memorize the given input producing a valid encoding.

    .. warnings::
      The array of labels is not used by the model since its function is just to encode the features.
      It is inserted in the function signature just for a compatibility with sklearn APIs.
    '''

    X = check_array(X)
    np.random.seed(self.seed)
    num_samples, num_features = X.shape

    if num_samples % self.batch_size != 0:
      raise ValueError('Minibatch size must be a divisor of the input size. Sorry, but this is a temporary solution.')

    self.weights = np.random.normal(loc=self.mu, scale=self.sigma, size=(self.outputs, num_features))
    self._fit(X, view=view)

    return self

  def _predict (self, X):
    '''
    Core function for the predict member
    '''
    #return self.activation(self.weights @ X.T)
    return self.activation(np.einsum('ij, kj -> ik', self.weights, X, optimize=True))

  def predict (self, X, y=None):
    '''
    Reduce X applying the Plasticity encoding.

    Parameters
    ----------
      X : array of shape (n_samples, n_features)
        The input samples

      y : array-like, default=None
        The array of labels

    Returns
    -------
      Xnew : array of shape (n_values, n_samples)
        The encoded features

    Notes
    -----
    .. warnings::
      The array of labels is not used by the model since its function is just to encode the features.
      It is inserted in the function signature just for a compatibility with sklearn APIs.
    '''
    check_is_fitted(self, 'weights')
    X = check_array(X)
    return self._predict(X)

  def transform (self, X):
    '''
    Apply the data reduction according to the features in the best signature found.

    Parameters
    ----------
      X : array-like of shape (n_samples, n_features)
        The input samples

    Returns
    -------
      Xnew : array-like of shape (n_samples, encoded_features)
        The data encoded according to the model weights.
    '''
    check_is_fitted(self, 'weights')
    Xnew = self._predict(X)
    return Xnew

  def fit_transform (self, X, y=None):
    '''
    Fit the model model meta-transformer and apply the data encoding transformation.

    Parameters
    ----------
      X : array-like of shape (n_samples, n_features)
        The training input samples

      y : array-like, shape (n_samples,)
        The target values

    Returns
    -------
      Xnew : array-like of shape (n_samples, encoded_features)
          The data encoded according to the model weights.

    Notes
    -----
    .. warnings::
      The array of labels is not used by the model since its function is just to encode the features.
      It is inserted in the function signature just for a compatibility with sklearn APIs.
    '''
    self.fit(X, y)
    Xnew = self.transform(X)
    return Xnew

  def save_weights (self, filename):
    '''
    Save the current weights to a binary file.

    Parameters
    ----------
      filename : str
        Filename or path

    Returns
    -------
      True if everything is ok
    '''
    check_is_fitted(self, 'weights')
    with open(filename, 'wb') as fp:
      self.weights.tofile(fp, sep='')
    return True

  def load_weights (self, filename):
    '''
    Load the weight matrix from a binary file.

    Parameters
    ----------
      filename : str
        Filename or path

    Returns
    -------
      self : object
        Return self
    '''
    with open(filename, 'rb') as fp:
      self.weights = np.fromfile(fp, dtype=np.float, count=-1)
    return self

  def __repr__(self):
    '''
    Object representation
    '''
    class_name = self.__class__.__qualname__
    params = self.__init__.__code__.co_varnames
    params = set(params) - {"self"}
    args = ", ".join(["{0}={1}".format(k, str(getattr(self, k))) for k in params])
    return "{0}({1})".format(class_name, args)
