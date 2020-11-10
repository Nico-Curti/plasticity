#ifndef __update_args_h__
#define __update_args_h__

#include <utils.hpp> // useful macros

#include <unordered_map> // std :: unordered_map
#include <numeric>       // std :: inner_product

#define ERROR_NWEIGHTS 201 ///< The weights dimension is incorrect

enum optimizer_t { _adam = 0, _momentum, _nesterov_momentum, _adagrad, _rmsprop, _adadelta, _adamax, _sgd
}; ///< optimizer types

namespace optimizer
{

static const std :: unordered_map < std :: string, int > get_optimizer {
                                                                          {"adam"              , _adam},
                                                                          {"momentum"          , _momentum},
                                                                          {"nesterov_momentum" , _nesterov_momentum},
                                                                          {"adagrad"           , _adagrad},
                                                                          {"rmsprop"           , _rmsprop},
                                                                          {"adadelta"          , _adadelta},
                                                                          {"adamax"            , _adamax},
                                                                          {"sgd"               , _sgd},
                                                                        }; ///< Utility for the optimizer management

}

/**
* @class update_args
* @brief Abstract type representing an optimization algorithm.
* The object implements different optimization algorithms, in
* particular:
*   - Adam
*   - Momentum
*   - NesterovMomentum
*   - AdaGrad
*   - RMSProp
*   - AdaDelta
*   - AdaMax
*   - SGD
*
* @details The desired optimization algorithm can be set using
* the type variable in the constructor signature.
* The core functionality of the object is given by the
* 'update' member function which applies the desired optimization
* algorithm step using the given parameters and gradients.
*
*/
class update_args
{

  static float epsil; ///< Numerical precision

protected:

  std :: unique_ptr < float [] > m; ///< Adam supporting array
  std :: unique_ptr < float [] > v; ///< Adam supporting array

  int nweights;        ///< Number of weights and thus the length of m and v arrays

public:

  int type;            ///< Optimization type to use

  float learning_rate; ///< Learning rate value
  float momentum;      ///< Momentum parameter
  float decay;         ///< Decay parameter
  float B1;            ///< Adam-like parameter
  float B2;            ///< Adam-like parameter
  float rho;           ///< TODO

  bool l2norm;         ///< Normalize the gradient values according to their l2 norms
  bool clip;           ///< Clip gradient values between -1 and 1

  // Constructors

  /**
  * @brief Default constructor.
  *
  */
  update_args ();

  /**
  * @brief Construct the object using the list of parameters.
  *
  * @details The constructor follows the same nomenclature of the Python counterpart.
  *
  * @note The type variable determines the desired optimizer.
  * The only difference between the Python counterpart is given by
  * the nweights parameter.
  *
  * @param type Optimization type to apply.
  * @param learning_rate Learning rate param.
  * @param momentum Learning rate momentum parameter.
  * @param decay Learning rate decay parameter.
  * @param B1 Adam parameter.
  * @param B2 Adam parameter.
  * @param rho TODO.
  * @param l2norm Switch if normalize the weights before update them.
  * @param clip Switch if clip the weights before update them.
  *
  */
  update_args (const int & type, float learning_rate=2e-2f, float momentum=.9f, float decay=1e-4f, float B1=.9f, float B2=.999f, float rho=0.f, bool l2norm=false, bool clip=false);

  // Destructors

  /**
  * @brief Destructor.
  *
  * @details Completely delete the object and release the memory of the arrays.
  *
  */
  ~update_args() = default;

  // Copy operators

  /**
  * @brief Copy operator.
  *
  * @details The operator performs a deep copy of the object and if there are buffers
  * already allocated, the operatore deletes them and then re-allocates an appropriated
  * portion of memory.
  *
  * @param args update_args object
  *
  */
  update_args & operator = (const update_args & args);

  /**
  * @brief Copy constructor.
  *
  * @details The copy constructor provides a deep copy of the object, i.e. all the
  * arrays are copied and not moved.
  *
  * @param args update_args object
  *
  */
  update_args (const update_args & args);


  /**
  * @brief Init the member arrays using the given number of weights
  *
  * @details This function init the member arrays used for the
  * optimization steps.
  *
  * @param nweights Number of weights/parameters to update.
  */
  void init_arrays (const int & nweights);

  /**
  * @brief Update the given parameters using the optimization algorithm
  *
  * @details This is the core functio of the object.
  *
  * @param iteration Current iteration number
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  * @param nweights Size of the given arrays.
  *
  */
  void update ( const int & iteration, float * weights, float * weights_update, const int & nweights );

private:

  /**
  * @brief Adam optimization step
  *
  * @param iteration Current iteration number
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void adam_update ( const int & iteration, float * weights, float * weights_update );

  /**
  * @brief Stochastic Gradient Descent optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void sgd_update ( float * weights, float * weights_update );

  /**
  * @brief Momentum optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void momentum_update ( float * weights, float * weights_update );

  /**
  * @brief Nesterov momentum optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void nesterov_momentum_update ( float * weights, float * weights_update );

  /**
  * @brief AdaDrad optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void adagrad_update ( float * weights, float * weights_update );

  /**
  * @brief RMSProp optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void rmsprop_update ( float * weights, float * weights_update );

  /**
  * @brief AdaDelta optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void adadelta_update ( float * weights, float * weights_update );

  /**
  * @brief AdaMax optimization step
  *
  * @param iteration Current iteration number
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void adamax_update ( const int & iteration, float * weights, float * weights_update );

  /**
  * @brief Normalize the array with l2 norm
  *
  * @param arr Input array
  * @param size Lenght of the given array
  *
  */
  void norm_value ( float * arr, const int & size );

  /**
  * @brief Clip the array values between [-1, 1]
  *
  * @param arr Input array
  * @param size Lenght of the given array
  *
  */
  void clip_value ( float * arr, const int & size );

};

#endif // __update_args_h__
