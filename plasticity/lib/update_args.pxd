# distutils: language = c++
# cython: language_level=2

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

cdef extern from "optimizer.h" nogil:

  cppclass update_args:

    update_args () except +
    update_args (const int & type, float learning_rate, float momentum, float decay, float B1, float B2, float rho, bool l2norm, bool clip) except +

    ## Attributes

    int type;

    float learning_rate;
    float momentum;
    float decay;
    float B1;
    float B2;
    float rho;

    bool l2norm;
    bool clip;

    ## Methods

    void init_arrays (const int & nweights)
    void update (const int & iteration, float * weights, float * weights_update, const int & nweights)


cdef extern from "<utility>" namespace "std" nogil:

  cdef unique_ptr[update_args] move(unique_ptr[update_args])

cdef class _update_args:

  cdef unique_ptr[update_args] thisptr
