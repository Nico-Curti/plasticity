#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pylab as plt
from inspect import isclass
from plasticity.utils import activations

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli5@unibo.it', 'nico.curti2@unibo.it']

def _check_activation (layer, activation_func):
  '''
  Check if the activation function is valid.

  Parameters
  ----------
  layer : object
    Layer object (ex. Activation_layer)

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
  You can use this function to verify if the given activation function is valid.
  The function can be passed either as a string either as object or simply as class object.

  Examples
  --------

  >>> layer = Activation_layer(input_shape=(1,2,3))
  >>> print(_check_activation(layer, 'Linear'))
      (6, 'Linear')
  >>> print(_check_activation(layer, Activations.Relu))
      (2, 'Relu')
  >>> print(_check_activation(layer, Activations.Linear()))
      (6, 'Linear')
  '''

  if isinstance(activation_func, str):
    allowed_activation_func = [f.lower() for f in dir(activations) if isclass(getattr(activations, f)) and f != 'Activations']

    if activation_func.lower() not in allowed_activation_func:
      class_name = layer.__class__.__name__
      raise ValueError('{0}: incorrect value of Activation Function given'.format(class_name))
    else:
      activation_func = activation_func.lower()
      activation_func = ''.join([activation_func[0].upper(), activation_func[1:]])

    activation_func = eval(''.join(['activations.', activation_func, '()']))
    activation = activation_func.ACTIVATION_INDEX

  elif issubclass(type(activation_func), activations.Activations):
    activation = activation_func.ACTIVATION_INDEX

  else:
    class_name = layer.__class__.__name__
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

