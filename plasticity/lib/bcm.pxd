# distutils: language = c++
# cython: language_level=2

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from update_args cimport update_args

cdef extern from "bcm.h" nogil:

  cppclass BCM:

    BCM (const int & outputs, const int & batch_size, const int & activation, update_args optimizer, float mu, float sigma, float interaction_strength, int seed) except +

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
