#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from collections import deque

from plasticity.utils import _check_activation
from plasticity.utils.activations import Linear
from plasticity.model.optimizer import Optimizer
from plasticity.model.weights import BaseWeights

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

__author__  = ['Nico Curti', 'Lorenzo Squadrani', 'SimoneGasperini']
__email__ = ['nico.curti2@unibo.it', 'lorenzo.squadrani@studio.unibo.it', 'simone.gasperini2@studio.unibo.it']
__all__ = ['BasePlasticity']

class BasePlasticity (BaseEstimator, TransformerMixin):

  '''
  Abstract base class for plasticity models

  Parameters
  ----------
    outputs : int (default=100)
      Number of hidden units.

    num_epochs : int (default=100)
      Maximum number of epochs for model convergence.

    batch_size : int (default=100)
      Size of the minibatch.

    weights_init : BaseWeights object
      Weights initialization strategy.

    activation : str (default="Linear")
      Name of the activation function.

    optimizer : Optimizer (default=SGD)
      Optimizer object (derived by the base class Optimizer).

    precision : float (default=1e-30)
      Parameter that controls numerical precision of the weight updates.

    epochs_for_convergence : int (default=None)
      Number of stable epochs requested for the convergence.
      If None the training proceeds up to the maximum number of epochs (num_epochs).

    convergence_atol : float (default=0.01)
      Absolute tolerance requested for the convergence.

    decay : float (default=0.)
      Weight decay scale factor.

    random_state : int (default=None)
      Random seed for batch subdivisions.

    verbose : bool (default=True)
      Turn on/off the verbosity.
  '''

  def __init__ (self, outputs : int = 100, num_epochs : int = 100,
      activation : str = 'Linear', optimizer : 'Optimizer' = Optimizer(),
      batch_size : int = 100, weights_init : 'BaseWeights' = BaseWeights(),
      precision : float = 1e-30,
      epochs_for_convergence : int = None,
      convergence_atol : float = 0.01,
      decay : float = 0.,
      random_state : int = None,
      verbose : bool = True):

    _, activation = _check_activation(self, activation)

    self.outputs = outputs
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.activation = activation
    self.optimizer = optimizer
    self.weights_init = weights_init
    self.precision = precision
    self.epochs_for_convergence = epochs_for_convergence if epochs_for_convergence is not None else num_epochs
    self.epochs_for_convergence = max(self.epochs_for_convergence, 1)
    self.decay = decay
    self.convergence_atol = convergence_atol
    self.random_state = random_state
    self.verbose = verbose

  def _weights_update (self, X : np.ndarray, output : np.ndarray) -> tuple:
    '''
    Compute the weights update using the given learning rule.

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

      theta : array-like
        Array of learning progress
    '''
    raise NotImplementedError

  def _lebesgue_norm (self, w : np.ndarray) -> np.ndarray:
    '''
    Apply the Lebesgue norm to the weights.

    Parameters
    ----------
      w : array-like (2D)
        Array to normalize using Lebesgue norm

    Returns
    -------
      wnorm : array-like (2D)
        Normalized version of the input array
    '''
    if self.p != 2:
      sign = np.sign(w)
      return sign * np.absolute(w)**(self.p - 1)
    else:
      return w

  def _fit_step (self, X : np.ndarray) -> np.ndarray:
    '''
    Core function of fit step (forward + backward + updates).
    We divide the step into a function to allow an easier visualization
    of the weight matrix (if necessary).

    Parameters
    ----------
      X : array-like (2D)
        Input array of data

    Returns
    -------
      theta : array-like
        Array of learning progress
    '''

    # predict the encoded values
    output = self._predict(X)

    # update weights
    w_update, theta = self._weights_update(X, output)

    # apply the weight decay
    if self.decay != 0.:
      w_update -= self.decay * self.weights

    #self.weights[:] += epsilon * w_update
    # -update for compatibility with optimizers
    self.weights, = self.optimizer.update(params=[self.weights], gradients=[-w_update])

    return theta

  @property
  def _check_convergence (self) -> bool:
    '''
    Check if the current training has reached the convergence.

    Returns
    -------
      check : bool
        Check if the learning history of the model is stable.

    Notes
    -----
    .. note::
      The convergence is estimated by the stability or not of the
      learning parameter in a fixed (epochs_for_convergence) number
      of epochs for all the outputs.
    '''

    if len(self.history) < self.epochs_for_convergence:
      return False

    last = np.full_like(self.history, fill_value=self.history[-1])

    return np.allclose(self.history, last, atol=self.convergence_atol)

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

  def _fit (self, X : np.ndarray) -> 'BasePlasticity':
    '''
    Core function for the fit member

    Parameters
    ----------
      X : array-like (2D)
        Input array of data

    Returns
    -------
      self
    '''

    num_samples, _ = X.shape
    indices = np.arange(0, num_samples).astype('int64')
    num_batches = num_samples // self.batch_size

    for epoch in range(self.num_epochs):

      if self.verbose:
        print('Epoch {:d}/{:d}'.format(epoch + 1, self.num_epochs))

      # random shuffle the input
      np.random.shuffle(indices)

      batches = np.lib.stride_tricks.as_strided(indices, shape=(num_batches, self.batch_size),
                                                strides=(self.batch_size * 8, 8))
      # init null values of theta for iterative summation
      theta = np.zeros(shape=(self.outputs, ), dtype=float)

      for batch in tqdm(batches, disable=(not self.verbose)):

        batch_data = X[batch, ...]

        theta += self._fit_step(X=batch_data)

      # append only the last theta value of the batch
      # for the convergence evaluation since the weight matrix
      # changes (it is updated) at every batch
      # Note: the theta must be normalized according to number of batches!
      self.history.append(theta * (1. / num_batches))

      # check if the model has reached the convergence (early stopping criteria)
      if self._check_convergence:
        if self.verbose:
          print('Early stopping: the training has reached the convergence criteria')
        break

      ##### WEIGHTS SYMMETRIC ORTHOGONALIZATION (once at the end of each epoch)
      # the two methods are actually exactly equivalent
      # (provable by simple linear algebra in real domain)
      # but the first using the numpy svd function is faster

      # (1)
      #U, _, Vt = np.linalg.svd(self.weights, full_matrices=False)
      #self.weights = np.einsum('ij, jk -> ik', U, Vt, optimize=True)

      # (2)
      #from scipy.linalg import sqrtm
      #self.weights = np.real(self.weights @ np.linalg.inv(sqrtm(self.weights.T @ self.weights)))

    return self

  def fit (self, X : np.ndarray, y : np.ndarray = None) -> 'BasePlasticity':
    '''
    Fit the Plasticity model weights.

    Parameters
    ----------
      X : array-like of shape (n_samples, n_features)
        The training input samples

      y : array-like, default=None
        The array of labels

    Returns
    -------
      self : object
        Return self

    Notes
    -----
    .. note::
      The model tries to memorize the given input producing a valid encoding.

    .. warning::
      If the array of labels is provided, it will be considered as a set of new inputs
      for the neurons. The labels can be 1D array or multi-dimensional array: the given
      shape is internally reshaped according to the required dimensions.
    '''

    if y is not None:
      X = self._join_input_label(X=X, y=y)

    X = check_array(X)
    np.random.seed(self.random_state)
    num_samples, num_features = X.shape

    if self.batch_size > num_samples:
      raise ValueError('Incorrect batch_size found. '
                       'The batch_size must be less or equal to the number of samples. '
                       'Given {:d} for {:d} samples'.format(self.batch_size, num_samples))

    #self.weights = np.random.normal(loc=self.mu,
    #                                scale=self.sigma,
    #                                size=(self.outputs, num_features))
    self.weights = self.weights_init.get(size=(self.outputs, num_features))
    self.history = deque(maxlen=self.epochs_for_convergence)
    self._fit(X)

    return self

  def _predict (self, X : np.ndarray) -> np.ndarray:
    '''
    Core function for the predict member
    '''
    raise NotImplementedError

  def predict (self, X : np.ndarray, y : np.ndarray = None) -> np.ndarray:
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
    .. warning::
      If the array of labels is provided, it will be considered as a set of new inputs
      for the neurons. The labels can be 1D array or multi-dimensional array: the given
      shape is internally reshaped according to the required dimensions.
    '''
    check_is_fitted(self, 'weights')

    if y is not None:
      X = self._join_input_label(X=X, y=y)

      # return (self.weights @ X).transpose()
      old_activation = self.activation
      self.activation = Linear()
      # without activation
      # np.einsum('ij, kj -> ik', self.weights, X, optimize=True).transpose()
      result = self._predict(X).transpose()
      self.activation = old_activation
      return result

    X = check_array(X)
    return self._predict(X)

  def transform (self, X : np.ndarray) -> np.ndarray:
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
    return Xnew.transpose()

  def fit_transform (self, X : np.ndarray, y : np.ndarray = None) -> np.ndarray:
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
    .. warning::
      If the array of labels is provided, it will be considered as a set of new inputs
      for the neurons. The labels can be 1D array or multi-dimensional array: the given
      shape is internally reshaped according to the required dimensions.
    '''
    self.fit(X, y=y)
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
    with open(filename, 'wb') as fp:
      self.weights.tofile(fp, sep='')
    return True

  def load_weights (self, filename : str) -> bool:
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

    # reshape the loaded weights since the numpy function loads
    # only in ravel format!!
    self.weights = self.weights.reshape(self.outputs, -1)

    return self

  def __repr__(self) -> str:
    '''
    Object representation
    '''
    class_name = self.__class__.__qualname__
    params = self.__init__.__code__.co_varnames
    params = set(params) - {'self'}
    args = ', '.join(['{0}={1}'.format(k, str(getattr(self, k)))
                      if not isinstance(getattr(self, k), str)
                      else '{0}="{1}"'.format(k, str(getattr(self, k)))
                      for k in params])
    return '{0}({1})'.format(class_name, args)
