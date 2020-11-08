# distutils: language = c++
# cython: language_level=2

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from update_args cimport update_args

cdef extern from "hopfield.h" nogil:

  cppclass Hopfield:

    Hopfield (const int & outputs, const int & batch_size, update_args optimizer, float mu, float sigma, float delta, float p, int k, int seed) except +

    ## Attributes

    ## Methods

    void fit (float * X, const int & n_samples, const int & n_features, const int & num_epochs)
    float * predict (const float * X, const int & n_samples, const int & n_features);

    void save_weights (const string & filename)
    void load_weights (const string & filename)

    float * get_weights ()

cdef extern from "<utility>" namespace "std" nogil:

  cdef unique_ptr[Hopfield] move(unique_ptr[Hopfield])

cdef class _Hopfield:

  cdef unique_ptr[Hopfield] thisptr

  cdef public:
    int outputs
    int n_features
