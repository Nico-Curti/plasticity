#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from plasticity.utils import _check_activation
from plasticity.utils import _check_string
from plasticity.utils import redirect_stdout
from .optimizer import Optimizer, SGD
from .weights import BaseWeights, Uniform

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

import numpy as np

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']
__all__ = ['BasePlasticity']

class BasePlasticity (BaseEstimator, TransformerMixin):

  '''
  Abstract base class for plasticity models

  Parameters
  ----------
    model : type
      Cython model type.

    outputs : int (default=100)
      Number of hidden units.

    num_epochs : int (default=100)
      Number of epochs for model convergence.

    batch_size : int (default=100)
      Size of the minibatch.

    activation : str (default="linear")
      Name of the activation function.

    optimizer : Optimizer object (default=SGD)
      The optimization algorithm to use during the training.

    weights_init : BaseWeights object (default="Uniform")
      Weights initialization strategy.

    epochs_for_convergence : int (default=None)
      Number of stable epochs requested for the convergence.
      If None the training proceeds up to the maximum
      number of epochs (num_epochs).

    convergence_atol : float (default=0.01)
      Absolute tolerance requested for the convergence.

    decay : float (default=0.)
      Weight decay scale factor.

    random_state : int (default=0)
      Random seed for weights generation.

    verbose : bool (default=True)
      Turn on/off the verbosity.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, model : object = None,
      outputs : int = 100, num_epochs : int = 100,
      batch_size : int = 100, activation : str = 'Linear',
      optimizer : 'Optimizer' = SGD(lr=2e-2),
      weights_init : 'BaseWeights' = Uniform(),
      epochs_for_convergence : int = None, convergence_atol : float = 0.01,
      decay : float = 0.,
      random_state : int = 0, verbose : bool = True,
      **kwargs):

    self.outputs = outputs
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.activation = activation
    self.optimizer = optimizer
    self.weights_init = weights_init
    self.epochs_for_convergence = epochs_for_convergence if epochs_for_convergence is not None else 1
    self.convergence_atol = convergence_atol
    self.decay = decay
    self.random_state = random_state
    self.verbose = verbose

    for k, v in kwargs.items():
      setattr(self, k, v)

    activation, _ = _check_activation(self, activation_func=activation)
    self._obj = model(self.outputs, self.batch_size, activation, self.optimizer._object,
                      self.weights_init._object, self.epochs_for_convergence,
                      self.convergence_atol,
                      self.decay, *kwargs.values())

  def _join_input_label (self, X : np.ndarray, y : np.ndarray) -> np.ndarray:
    '''
    Join the input data matrix to the labels.
    In this way the labels array/matrix is considered as a new
    set of inputs for the model and the plasticity model can
    perform classification tasks without any extra supervised learning.

    Parameters
    ----------
      X : array-like (2D)
        Input array of data

      y : array-like (1D or 2D)
        Labels array/matrix

    Returns
    -------
      join : array-like (2D)
        Matrix of the merged data in which the first n_sample columns
        are occupied by the original data and the remaining ones store
        the labels.

    Notes
    -----
    .. note::
      The labels can be a 1D array or multi-dimensional array: the given
      shape is internally reshaped according to the required dimensions.
    '''

    X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
    # reshape the labels if it is a single array
    y = y.reshape(-1, 1) if len(y.shape) == 1 else y
    # concatenate the labels as new inputs for neurons
    X = np.concatenate((X, y), axis=1)

    return X

  def fit (self, X : np.ndarray, y : np.ndarray = None) -> 'BasePlasticity':
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
      The array of labels is not used by the model since its function is just
      to encode the features.
      It is inserted in the function signature just for a compatibility
      with sklearn APIs.
    '''

    if y is not None:
      X = self._join_input_label(X=X, y=y)

    X = check_array(X)
    num_samples, num_features = X.shape
    X = np.ascontiguousarray(X.ravel().astype('float32'))

    if self.batch_size > num_samples:
      raise ValueError('Incorrect batch_size found. '
                       'The batch_size must be less or equal to the number of samples. '
                       'Given {:d} for {:d} samples'.format(self.batch_size, num_samples))

    with redirect_stdout(self.verbose):
      self._obj.fit(X, num_samples, num_features, self.num_epochs, self.random_state)

    self.weights = self._obj.get_weights()

    return self

  def predict (self, X : np.ndarray, y : np.ndarray = None) -> np.ndarray:
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
      The array of labels is not used by the model since its function is
      just to encode the features.
      It is inserted in the function signature just for a compatibility
      with sklearn APIs.
    '''
    check_is_fitted(self, 'weights')

    if y is not None:
      X = self._join_input_label(X=X, y=y)

      # TODO: implement prediction without activation
      return np.einsum('ij, kj -> ik', self.weights, X, optimize=True).transpose()

    X = check_array(X)
    num_samples, num_features = X.shape

    X = np.ascontiguousarray(X.ravel().astype('float32'))

    output = self._obj.predict(X, num_samples, num_features)
    return np.asarray(output).reshape(self._obj.outputs, num_samples)

  def transform (self, X : np.ndarray) -> np.ndarray:
    '''
    Apply the data reduction according to the features in the best
    signature found.

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
    return Xnew.transpose()

  def fit_transform (self, X : np.ndarray, y : np.ndarray = None) -> np.ndarray:
    '''
    Fit the model model meta-transformer and apply the data
    encoding transformation.

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
      The array of labels is not used by the model since its function is
      just to encode the features.
      It is inserted in the function signature just for a compatibility
      with sklearn APIs.
    '''
    self.fit(X, y)
    Xnew = self.transform(X)
    return Xnew

  def save_weights (self, filename : str) -> bool:
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

    filename = _check_string(filename, exist=False)
    self._obj.save_weights(filename)

    return True

  def load_weights (self, filename : str) -> 'BasePlasticity':
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

  def __repr__ (self) -> str:
    '''
    Object representation
    '''
    class_name = self.__class__.__qualname__
    params = self.__init__.__code__.co_varnames
    params = set(params) - {'self', 'kwargs', 'model'}
    args = ', '.join(['{0}={1}'.format(k, str(getattr(self, k)))
                      if not isinstance(getattr(self, k), str)
                      else '{0}="{1}"'.format(k, str(getattr(self, k)))
                      for k in params])
    return '{0}({1})'.format(class_name, args)
