#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pylab as plt
from inspect import isclass
from plasticity.utils import activations

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli5@unibo.it', 'nico.curti2@unibo.it']

def _check_activation (obj, activation_func):
  '''
  Check if the activation function is valid.

  Parameters
  ----------
    obj : object
      Object type which call this function

    activation_func : string or Activations object
      activation function to check. If the Activations object is not created yet
      the 'eval' is done on the object.

  Returns
  -------
    index : int
      C++ activation function index
    name : str
      Activation function name

  Notes
  -----
  .. note::
    You can use this function to verify if the given activation function is valid.
    The function can be passed either as a string either as object or simply as class object.

  Examples
  --------
  >>> obj = BCM()
  >>> print(_check_activation(obj, 'Linear'))
      (6, 'Linear')
  >>> print(_check_activation(obj, Activations.Relu))
      (2, 'Relu')
  >>> print(_check_activation(obj, Activations.Linear()))
      (6, 'Linear')
  '''

  if isinstance(activation_func, str):
    allowed_activation_func = [f.lower() for f in dir(activations) if isclass(getattr(activations, f)) and f != 'Activations']

    if activation_func.lower() not in allowed_activation_func:
      class_name = obj.__class__.__name__
      raise ValueError('{0}: incorrect value of Activation Function given'.format(class_name))
    else:
      activation_func = activation_func.lower()
      activation_func = ''.join([activation_func[0].upper(), activation_func[1:]])

    activation_func = eval(''.join(['activations.', activation_func, '()']))
    activation = activation_func.ACTIVATION_INDEX

  elif issubclass(type(activation_func), activations.Activations):
    activation = activation_func.ACTIVATION_INDEX

  else:
    class_name = obj.__class__.__name__
    raise ValueError('{0}: incorrect value of Activation Function given'.format(class_name))

  # temporary solution to avoid not implemented activation function
  if activation > 13: # Selu is the last available activation function
    raise NotImplementedError('The {0} activation function is not implemented yet!'.format(activation_func.name))

  return (activation, activation_func)

def _check_string (string, exist=True):
  '''
  Check if the input string is already encoded for c++ compatibility

  Parameters
  ----------
    string : string or bytes
      string to convert / verify

    exist : bool (default = True)
      If the string identify a filename check if it exist

  Returns
  -------
    res: bytes
      Encoded string (utf-8)

  Notes
  -----
  The strings must be converted to bytes for c++ function compatibility!
  '''

  if not isinstance(string, str) and not isinstance(string, bytes):
    raise TypeError('{} must be in string format'.format(string))

  if exist:
    if not os.path.isfile(string):
      raise FileNotFoundError('Could not open or find the data file. Given: {}'.format(string))

  return string.encode('utf-8') if isinstance(string, str) else string

def _check_update (upd_type):
  '''
  Check if the update function is valid.

  Parameters
  ----------
    upd_type : string or int
      update function to check.

  Returns
  -------
    update_type : str
      Name of the update function

    update_num : int
      Byron update function index

  Notes
  -----
  .. note::
    You can use this function to verify if the given update function is valid.
    The function can be passed either as a string either as integer.

  Examples
  --------
  >>> name, type = _check_update('Adam')
  >>> print(name, type)
      ('Adam', 0)
  >>> print(_check_update(2))
      ('Nesterov-Momentum', 2)
  '''

  allowed_upd = ('Adam', 'Momentum', 'NesterovMomentum', 'Adagrad', 'RMSprop', 'Adadelta', 'Adamax', 'SGD')

  if isinstance(upd_type, str):

    if upd_type not in allowed_upd:
      raise ValueError('Optimizer: incorrect value of Update Function given. Possible values are ({})'.format(', '.join(allowed_upd)))

    update_type = upd_type
    update_num = allowed_upd.index(update_type)

  elif isinstance(upd_type, int) and upd_type < len(allowed_upd):
    update_type = allowed_upd[upd_type]
    update_num = upd_type

  else:
    raise ValueError('Optimizer: incorrect value of Update Function given. Possible values are ({})'.format(', '.join(allowed_upd)))

  return (update_type, update_num)



def view_weights (weights, dims):
  '''
  Plot the weight matrix as full image

  Parameters
  ----------
    weights : array-like
      Weight matrix as (num_outputs, num_features)

    dims : tuple
      Dimension of each single image/weight connections

  Returns
  -------
    None

  Example
  -------
  >>> from plasiticy.model import BCM
  >>> from plasticity.utils import view_weights
  >>>
  >>> model = BCM(outputs=100, num_epochs=10, batch_size=100, activation='relu',
                  optimizer=Adam(lr=2e-2), interaction_strength=0.)
  >>> model.fit(X)
  >>> view_weights (model.weights, dims=(28, 28))
  '''

  num_images = int(np.sqrt(weights.shape[0]))

  # extract the maximum number of weights for a square image
  selected_weights = weights[:num_images**2]

  # combine the series of images into a full matrix
  image = np.hstack(np.hstack(selected_weights.reshape(num_images, num_images, *dims)))

  # colormap range
  nc = np.amax(np.abs(selected_weights))

  # plot the results
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
  im = ax.imshow(image, cmap='bwr', vmin=-nc, vmax=nc)
  fig.colorbar(im, ticks=[np.amin(selected_weights), 0, np.amax(selected_weights)])

  plt.show()

