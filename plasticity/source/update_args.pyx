# distutils: language = c++
# cython: language_level=2

from cython.operator cimport dereference as deref

from update_args cimport update_args

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']


cdef class _update_args:

  def __init__(self, int type, float learning_rate,
    float momentum, float decay,
    float B1, float B2, float rho):
    self.thisptr.reset(new update_args(type, learning_rate, momentum, decay, B1, B2, rho))

  @property
  def get_learning_rate(self):
    return deref(self.thisptr).learning_rate

  @property
  def get_momentum(self):
    return deref(self.thisptr).momentum

  @property
  def get_decay(self):
    return deref(self.thisptr).decay

  @property
  def get_B1(self):
    return deref(self.thisptr).B1

  @property
  def get_B2(self):
    return deref(self.thisptr).B2

  @property
  def get_rho(self):
    return deref(self.thisptr).rho
