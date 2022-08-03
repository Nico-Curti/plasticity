#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import numpy as np
import pylab as plt
from inspect import isclass
from contextlib import contextmanager
from plasticity.utils import activations

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli5@unibo.it', 'nico.curti2@unibo.it']

__all__ = ['_check_activation', '_check_string', '_check_update',
           '_check_weights_init', 'redirect_stdout',
           'view_weights',
           ]

def _check_activation (obj : object, activation_func : str) -> tuple:
  '''
  Check if the activation function is valid.

  Parameters
  ----------
    obj : object
      Object type which call this function

    activation_func : string or Activations object
      activation function to check. If the Activations object is
      not created yet the 'eval' is done on the object.

  Returns
  -------
    index : int
      C++ activation function index
    name : str
      Activation function name

  Notes
  -----
  .. note::
    You can use this function to verify if the given activation
    function is valid.
    The function can be passed either as a string either as object
    or simply as class object.

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
    allowed_activation_func = [f.lower() for f in dir(activations)
                               if isclass(getattr(activations, f)) and f != 'Activations']

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
  if activation > 18: # Asymmetric Logistic is the last available activation function
    raise NotImplementedError('The {0} activation function is not implemented yet!'.format(
      activation_func.name))

  return (activation, activation_func)

def _check_string (string : str, exist : bool = True) -> bytes:
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
      raise FileNotFoundError('Could not open or find the data file. Given: {}'.format(
        string))

  return string.encode('utf-8') if isinstance(string, str) else string

def _check_update (upd_type : str) -> tuple:
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
      C++ update function index

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
      raise ValueError('Optimizer: incorrect value of Update Function given. Possible values are ({})'.format(
        ', '.join(allowed_upd)))

    update_type = upd_type
    update_num = allowed_upd.index(update_type)

  elif isinstance(upd_type, int) and upd_type < len(allowed_upd):
    update_type = allowed_upd[upd_type]
    update_num = upd_type

  else:
    raise ValueError('Optimizer: incorrect value of Update Function given. Possible values are ({})'.format(
      ', '.join(allowed_upd)))

  return (update_type, update_num)

def _check_weights_init (init_type : str) -> tuple:
  '''
  Check if the weights initialization function is valid.

  Parameters
  ----------
    init_type : string or int
      weights initialization function to check.

  Returns
  -------
    init_type : str
      Name of the weights initialization function

    init_num : int
      C++ weights initialization function index

  Notes
  -----
  .. note::
    You can use this function to verify if the given weights initialization function is valid.
    The function can be passed either as a string either as integer.

  Examples
  --------
  >>> name, type = _check_weights_init('Uniform')
  >>> print(name, type)
      ('Uniform', 2)
  >>> print(_check_weights_init(2))
      ('Uniform', 2)
  '''

  allowed_init = ('Zeros', 'Ones', 'Uniform', 'Normal', 'LecunUniform', 'GlorotUniform', 'GlorotNormal', 'HeUniform', 'HeNormal')

  if isinstance(init_type, str):

    if init_type not in allowed_init:
      raise ValueError('Weights initialization: incorrect value of weights initialization Function given. Possible values are ({})'.format(
        ', '.join(allowed_init)))

    init_type = init_type
    init_num = allowed_init.index(init_type)

  elif isinstance(init_type, int) and init_type < len(allowed_init):
    init_type = allowed_init[init_type]
    init_num = init_type

  else:
      raise ValueError('Weights initialization: incorrect value of weights initialization Function given. Possible values are ({})'.format(
        ', '.join(allowed_init)))

  return (init_type, init_num)


@contextmanager
def redirect_stdout (verbose : bool):
  '''
  Redirect output stdout from cython wrap to devnull or not.
  This function works ONLY for cython wrap functions!!
  If you want to redirect python prints you can use something like

  Parameters
  ----------
    verbose: bool
      Switch if turn on/off the output redirection

  Example
  -------
  >>> from io import StringIO
  >>> import contextlib
  >>> temp_stdout = StringIO()
  >>> foo = lambda : print('hello world!')
  >>> with contextlib.redirect_stdout(temp_stdout):
  >>>   foo()

  '''

  try:
    # Temporary fix for the IPython console.
    # The current version of the redirect_stdout does not support the
    # redirection using IPython.
    # Error: "ValueError: write to closed file"
    # TODO: fix this issue with some workaround for the devnull redirection in IPython
    __IPYTHON__
    warnings.warn('The current version does not allow to redirect the stdout using an IPython console.', RuntimeWarning)
    verbose = True

  except NameError:
    pass

  if verbose:
    try:
      yield
    finally:
      return

  to = os.devnull

  fd = sys.stdout.fileno()

  # assert that Python and C stdio write using the same file descriptor
  # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

  def _redirect_stdout (to):
    sys.stdout.close()              # + implicit flush()
    os.dup2(to.fileno(), fd)        # fd writes to 'to' file
    sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

  with os.fdopen(os.dup(fd), 'w') as old_stdout:
    with open(to, 'w') as file:
      _redirect_stdout(to=file)
    try:
      yield # allow code to be run with the redirected stdout
    finally:
      _redirect_stdout(to=old_stdout) # restore stdout.
                                      # buffering and flags such as
                                      # CLOEXEC may be different


def view_weights (weights : np.ndarray, dims : tuple) -> None:
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
  nc = np.max(np.abs(selected_weights))

  # plot the results
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
  ax.axis('off')
  im = ax.imshow(image, cmap='bwr', vmin=-nc, vmax=nc)
  fig.colorbar(im, ticks=[np.min(selected_weights), 0, np.max(selected_weights)])

  plt.show()
