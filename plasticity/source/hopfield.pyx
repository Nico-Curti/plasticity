# distutils: language = c++
# cython: language_level=2

from libcpp.string cimport string
from cython.operator cimport dereference as deref

from hopfield cimport Hopfield
from update_args cimport _update_args
from weights_initialization cimport _weights_initialization

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']


cdef class _Hopfield:

  def __init__ (self, int outputs, int batch_size, int activation,
    _update_args optimizer, _weights_initialization w_init,
    int epochs_for_convergence,
    float convergence_atol, float decay,
    float delta, float p, int k):

    self.thisptr.reset(new Hopfield(outputs, batch_size,
                                    deref(optimizer.thisptr.get()), deref(w_init.thisptr.get()),
                                    epochs_for_convergence,
                                    convergence_atol, decay,
                                    delta, p, k))
    self.outputs = outputs
    self.n_features = 0

  def fit (self, float[::1] X, int n_samples, int n_features, int num_epochs, int seed):

    self.n_features = n_features
    deref(self.thisptr).fit(&X[0], n_samples, n_features, num_epochs, seed)

  def predict (self, float[::1] X, int n_samples, int n_features):

    cdef float * res = deref(self.thisptr).predict(&X[0], n_samples, n_features)
    return [res[i] for i in range(self.outputs * n_samples)]

  def get_weights (self):

    if self.n_features == 0:
      return (None, None)

    cdef float * w = deref(self.thisptr).get_weights()
    return ([w[i] for i in range(self.outputs * self.n_features)], (self.outputs, self.n_features))

  def save_weights (self, string filename):
    deref(self.thisptr).save_weights(filename)

  def load_weights (self, string filename):
    deref(self.thisptr).load_weights(filename)
