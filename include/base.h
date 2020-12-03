#ifndef __base_h__
#define __base_h__

#include <activations.h>
#include <optimizer.h>
#include <weights.h>
#include <utils.hpp>

#include <memory>
#include <deque>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <utility>

#include <iostream>

#define ERROR_K_POSITIVE 101
#define ERROR_FITTED     102
#define ERROR_DIMENSIONS 103
#define ERROR_CONVERGENCY_POSITIVE 104

/**
* @class BasePlasticity
* @brief Abstract type representing an encoder model, i.e. a neural network
* ables to memorize all the input data giving in output an encoding array of
* features for each input data.
*
* @details This class is the base class for specialized models.
* The derived classes have to implement an appropriated version of
* the private member function "weights_update", i.e. the function
* responsibles for the update of the weights matrix.
* A second member which could be specialized is the "normalize_weights"
* private member which is responsible of the normalization of the weights
* matrix **before** the fit function.
*
*/
class BasePlasticity
{

protected:

  update_args optimizer;                 ///< optimizer object
  weights_initialization w_init;         ///< weights initialization object

  std :: unique_ptr < float[] > output;  ///< array of outputs

#ifdef __testing__
public:
#endif

  std :: unique_ptr < float[] > weights; ///< array-matrix of weights

#ifdef __testing__
protected:
#endif

  std :: deque < std :: unique_ptr < float[] > > history; ///< deque for the convergency monitoring
  std :: unique_ptr < float[] > theta; ///< array of means

  std :: function < float(const float &) > activation; ///< pointer to activation function
  std :: function < float(const float &) > gradient;   ///< pointer to gradient function

protected:

  static float precision; ///< Parameter that controls numerical precision of the weight updates.

  int batch;                  ///< batch size
  int outputs;                ///< number of hidden units
  int nweights;               ///< number of weights
  int epochs_for_convergency; ///< number of stable epochs requested for the convergency

  float convergency_atol; ///< Absolute tolerance requested for the convergency

public:

  // Constructor


  /**
  * @brief Default constructor.
  *
  */
  BasePlasticity ();

  /**
  * @brief Construct the object using the list of training parameters.
  *
  * @details The constructor follows the same nomenclature of the Python counterpart.
  * This is the abstract type for the plasticity model.
  *
  * @note Overriding this class you can specify the weights-update rule to use in the training.
  *
  * @param outputs Number of hidden units.
  * @param batch_size Size of the minibatch.
  * @param activation Index of the activation function.
  * @param optimizer update_args Optimizer object (default=SGD algorithm).
  * @param weights_init weights_initialization object (default=uniform initialization in [-1, 1]).
  * @param epochs_for_convergency Number of stable epochs requested for the convergency.
  * @param convergency_atol Absolute tolerance requested for the convergency.
  *
  */
  BasePlasticity (const int & outputs, const int & batch_size, int activation=transfer :: _linear_,
                  update_args optimizer=update_args(optimizer_t :: _sgd),
                  weights_initialization weights_init=weights_initialization(weights_init_t :: _uniform_),
                  int epochs_for_convergency=1, float convergency_atol=1e-2f);


  // Copy Operator and Copy Constructor

  /**
  * @brief Copy constructor.
  *
  * @details The copy constructor provides a deep copy of the object, i.e. all the
  * arrays are copied and not moved.
  *
  * @param b BasePlasticity object
  *
  */
  BasePlasticity (const BasePlasticity & b);

  /**
  * @brief Copy operator.
  *
  * @details The operator performs a deep copy of the object and if there are buffers
  * already allocated, the operatore deletes them and then re-allocates an appropriated
  * portion of memory.
  *
  * @param b BasePlasticity object
  *
  */
  BasePlasticity & operator = (const BasePlasticity & b);

  // Destructor

  /**
  * @brief Destructor.
  *
  * @details Completely delete the object and release the memory of the arrays.
  *
  */
  ~BasePlasticity () = default;

  // Public members

  /**
  * @brief Train the model/encoder
  *
  * @details The model computes the weights and thus the encoded features
  * using the given plasticity rule.
  * The signature of the function is totally equivalent to the the Python counterpart
  * except by the pointer arrays which require the dimension size as extra parameters.
  *
  * @note This function must be called before the predict member-function.
  * A check is performed internally to ensure it.
  *
  * @param X array in ravel format of the input variables/features
  * @param n_samples dimension of the X matrix, i.e. the number of rows
  * @param n_features dimension of the X matrix, i.e. the number of cols
  * @param num_epochs Number of epochs for model convergency.
  * @param seed Random seed number for the batch subdivisions.
  *
  */
  void fit (float * X, const int & n_samples, const int & n_features, const int & num_epochs, int seed=42);

  /**
  * @brief Train the model/encoder
  *
  * @details The model computes the weights and thus the encoded features
  * using the given plasticity rule.
  * The signature of the function is totally equivalent to the the Python counterpart
  * except by the pointer arrays which require the dimension size as extra parameters.
  *
  * @note This function must be called before the predict member-function.
  * A check is performed internally to ensure it.
  *
  * @param X array in ravel format of the input variables/features.
  * @param n_samples dimension of the X matrix, i.e. the number of rows.
  * @param n_features dimension of the X matrix, i.e. the number of cols.
  *
  * @return The array of encoded features.
  *
  */
  float * predict (float * X, const int & n_samples, const int & n_features);

  /**
  * @brief Save the current weight matrix.
  *
  * @details The weights matrix is saved in binary format.
  * The first value of the file is an integer corresponding to the number of
  * weights (rows x cols) of the weight matrix, followed by the (float) weight matrix
  * in ravel format.
  *
  * @param filename Filename or path where the file is saved.
  *
  */
  void save_weights (const std :: string & filename);

  /**
  * @brief Load the current weight matrix.
  *
  * @details The weights matrix is loaded according to the format
  * specified in the save_weights function, i.e. the first first value of the file
  * is an integer corresponding to the number of weights (rows x cols) of the weight matrix,
  * followed by the (float) weight matrix
  * in ravel format.
  *
  * @param filename Filename or path of the weight.
  */
  void load_weights (const std :: string & filename);

  /**
  * @brief Get the weight matrix as pointer array
  *
  * @details This function is just an utility for the Cython wrap
  * of the object.
  *
  * @return The weights matrix in ravel format.
  */
  float * get_weights ();

private:

  /**
  * @brief Weights update rule.
  *
  * @note Compute the weights update using the given learning rule.
  *
  * @param X array in ravel format of the input variables/features.
  * @param n_features dimension of the X matrix, i.e. the number of cols.
  * @param weights_update Array/matrix of updates for weights.
  *
  */
  virtual void weights_update (float * X, const int & n_features, float * weights_update) = 0;

  /**
  * @brief Check the input dimensions.
  *
  * @note The function checks if the given dimensions are consistent with the input ones.
  *
  * @param n_features dimension of the X matrix, i.e. the number of cols
  *
  */
  void check_dims (const int & n_features);

  /**
  * @brief Check if the model is already fitted.
  *
  * @note The function checks if function fit has been already called before the prediction.
  * The check is performed on the value of the output array
  *
  */
  void check_is_fitted ();

  /**
  * @brief Check the given parameters.
  *
  * @note The function checks if the input variable epochs_for_convergency is positive defined and greater than 1
  *
  */
  void check_params ();

  /**
  * @brief Check if the model training has reached the convergency.
  *
  * @note The convergency is estimated by the stability or not of the
  * learning parameter in a fixed (epochs_for_convergency) number
  * of epochs for all the outputs.
  *
  */
  bool check_convergency ();

  /**
  * @brief Normalize the weights according to the given function.
  *
  * @note This function must be overrided.
  *
  */
  virtual void normalize_weights ();

  /**
  * @brief Core function of the fit formula
  *
  * @note
  *
  * @param X array in ravel format of the input variables/features.
  * @param num_epochs Number of epochs for model convergency.
  * @param n_samples dimension of the X matrix, i.e. the number of rows
  * @param n_features dimension of the X matrix, i.e. the number of cols
  * @param seed Random seed number for the batch subdivisions.
  *
  */
  void _fit (float * X, const int & num_epochs, const int & n_features, const int & n_samples, const int & seed);

  /**
  * @brief Core function of the predict formula
  *
  * @note The function computes the output as W @ X.T.
  * We use the GEMM algorithm with OpenMP support for a fast evaluation
  *
  * @param A Input matrix (M x K)
  * @param B Input matrix (N x K)
  * @param C Output matrix (M x N)
  * @param N Number of rows of B
  * @param M Number of rows of A
  * @param K Number of cols of A and B
  * @param buffer extra array buffer with shape (M x K) to use if necessary
  *
  */
  virtual void _predict (float * A, float * B, float * C, const int & N, const int & M, const int & K, float * buffer=nullptr);

};


#endif // __base_h__
