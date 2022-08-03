# distutils: language = c++
# cython: language_level=2

from cython.operator cimport dereference as deref

from weights_initialization cimport weights_initialization

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']


cdef class _weights_initialization:

  def __init__(self, int type, float mu, float sigma, float scale, int seed):
    self.thisptr.reset(new weights_initialization(type, mu, sigma, scale, seed))
