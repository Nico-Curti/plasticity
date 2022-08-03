# distutils: language = c++
# cython: language_level=2

from libcpp.string cimport string
from cython.operator cimport dereference as deref

cimport numpy as np
import numpy as np

from bcm cimport BCM
from update_args cimport _update_args
from weights_initialization cimport _weights_initialization

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

cdef class _BCM:

  def __init__ (self, int outputs, int batch_size, int activation,
    _update_args optimizer, _weights_initialization w_init,
    int epochs_for_convergence,
    float convergence_atol, float decay,
    float memory_factor, float interaction_strenght):

    self.thisptr.reset(new BCM(outputs, batch_size, activation,
                               deref(optimizer.thisptr.get()), deref(w_init.thisptr.get()),
                               epochs_for_convergence,
                               convergence_atol, decay,
                               memory_factor, interaction_strenght))
    self.outputs = outputs
    self.n_features = 0

  def fit (self, float[::1] X, int n_samples, int n_features, int num_epochs, int seed):

    self.n_features = n_features
    deref(self.thisptr).fit(&X[0], n_samples, n_features, num_epochs, seed)

  def predict (self, float[::1] X, int n_samples, int n_features):

    cdef float * res = deref(self.thisptr).predict(&X[0], n_samples, n_features)
    return np.asarray(<np.float32_t[: self.outputs * n_samples]> res)

  def get_weights (self):

    if self.n_features == 0:
      return (None, None)

    cdef float * w = deref(self.thisptr).get_weights()
    weights = np.asarray(<np.float32_t[:self.outputs * self.n_features]> w)
    return weights.reshape(self.outputs, self.n_features)

  def save_weights (self, string filename):
    deref(self.thisptr).save_weights(filename)

  def load_weights (self, string filename):
    deref(self.thisptr).load_weights(filename)
