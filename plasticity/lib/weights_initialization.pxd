# distutils: language = c++
# cython: language_level=2

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

cdef extern from "weights.h" nogil:

  cppclass weights_initialization:

    weights_initialization () except +
    weights_initialization (const int & type, float mu, float sigma, float scale, int seed) except +

    ## Attributes

    int type;

    float mu;
    float sigma;
    float scale;


cdef extern from "<utility>" namespace "std" nogil:

  cdef unique_ptr[weights_initialization] move(unique_ptr[weights_initialization])

cdef class _weights_initialization:

  cdef unique_ptr[weights_initialization] thisptr
