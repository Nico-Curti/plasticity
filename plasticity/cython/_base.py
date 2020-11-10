#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from plasticity.utils import _check_activation
from plasticity.utils import _check_string
from .optimizer import Optimizer

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

import numpy as np

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


class BasePlasticity (BaseEstimator, TransformerMixin):

  '''
  Abstract base class for plasticity models

  Parameters
  ----------
    model : type
      Cython model type

    outputs : int (default=100)
      Number of hidden units

    num_epochs : int (default=100)
      Number of epochs for model convergency

    batch_size : int (default=100)
      Size of the minibatch

    activation : str (default="linear")
      Name of the activation function

    optimizer : Optimizer object
      The optimization algorithm to use during the training

    mu : float (default=0.)
      Mean of the gaussian distribution that initializes the weights

    sigma : float (default=1.)
      Standard deviation of the gaussian distribution that initializes the weights

    seed : int (default=42)
      Random seed for weights generation

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, model=None, outputs=100, num_epochs=100,
      batch_size=100, activation='Linear', optimizer=Optimizer(update_type='SGD'),
      mu=0., sigma=1.,
      seed=42, **kwargs):

    self.outputs = outputs
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.activation = activation
    self.optimizer = optimizer
    self.mu = mu
    self.sigma = sigma
    self.seed = seed

    for k, v in kwargs.items():
      setattr(self, k, v)

    activation, _ = _check_activation(self, activation_func=activation)
    self._obj = model(outputs, batch_size, activation, optimizer._object,
                      mu, sigma, seed, *kwargs.values())

  def fit (self, X, y=None):
    '''
    Fit the Plasticity model weights.

    Parameters
    ----------
      X : array-like of shape (n_samples, n_features)
        The training input samples

      y : array-like (default=None)
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
    num_samples, num_features = X.shape
    X = np.ascontiguousarray(X.ravel().astype('float32'))

    self._obj.fit(X, num_samples, num_features, self.num_epochs)

    self.weights, shape = self._obj.get_weights()
    self.weights = np.asarray(self.weights).reshape(shape)

    #return self

  def predict (self, X, y=None):
    '''
    Reduce X applying the Plasticity encoding.

    Parameters
    ----------
      X : array of shape (n_samples, n_features)
        The input samples

      y : array-like (default=None)
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
    num_samples, num_features = X.shape

    X = np.ascontiguousarray(X.ravel().astype('float32'))

    output = self._obj.predict(X, num_samples, num_features)
    return np.asarray(output).reshape(self._obj.outputs, num_samples)

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
    Xnew = self.predict(X)
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

    filename = _check_string(filename, exist=True)
    self._obj.save_weights(filename)

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

    filename = _check_string(filename, exist=True)
    self._obj.load_weights(filename)

    return self

  def __repr__ (self):
    '''
    Object representation
    '''
    class_name = self.__class__.__qualname__
    params = self.__init__.__code__.co_varnames
    params = set(params) - {'self', 'kwargs', 'model'}
    args = ', '.join(['{0}={1}'.format(k, str(getattr(self, k)))
                      if not isinstance(getattr(self, k), str) else '{0}="{1}"'.format(k, str(getattr(self, k)))
                      for k in params])
    return '{0}({1})'.format(class_name, args)
