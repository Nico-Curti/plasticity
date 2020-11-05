# distutils: language = c++
# cython: language_level=2

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

cdef extern from "bcm.h" nogil:

  cppclass BCM:

    BCM (const int & outputs, const int & batch_size, const int & activation, float mu, float sigma, float epsilon, float interaction_strenght, int seed) except +

    ## Attributes

    ## Methods

    void fit (float * X, const int & n_samples, const int & n_features, const int & num_epochs)
    float * predict (const float * X, const int & n_samples, const int & n_features);

    void save_weights (const string & filename)
    void load_weights (const string & filename)

    float * get_weights ()

cdef extern from "<utility>" namespace "std" nogil:

  cdef unique_ptr[BCM] move(unique_ptr[BCM])

cdef class _BCM:

  cdef unique_ptr[BCM] thisptr

  cdef public:
    int outputs
    int n_features
