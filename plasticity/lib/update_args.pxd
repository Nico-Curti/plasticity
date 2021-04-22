# distutils: language = c++
# cython: language_level=2

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

cdef extern from "optimizer.h" nogil:

  cppclass update_args:

    update_args () except +
    update_args (const int & type, float learning_rate, float momentum, float decay, float B1, float B2, float rho) except +

    ## Attributes

    int type;

    float learning_rate;
    float momentum;
    float decay;
    float B1;
    float B2;
    float rho;



cdef extern from "<utility>" namespace "std" nogil:

  cdef unique_ptr[update_args] move(unique_ptr[update_args])

cdef class _update_args:

  cdef unique_ptr[update_args] thisptr
