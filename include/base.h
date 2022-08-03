/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  The OpenHiP package is licensed under the MIT "Expat" License:
//
//  Copyright (c) 2021: Nico Curti.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  the software is provided "as is", without warranty of any kind, express or
//  implied, including but not limited to the warranties of merchantability,
//  fitness for a particular purpose and noninfringement. in no event shall the
//  authors or copyright holders be liable for any claim, damages or other
//  liability, whether in an action of contract, tort or otherwise, arising from,
//  out of or in connection with the software or the use or other dealings in the
//  software.
//
//M*/

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

#include <Eigen/Dense>

#if EIGEN_VERSION_AT_LEAST(3, 3, 90)

  #include <Eigen/Core> // Eigen :: all slicing

#endif


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

public:

  Eigen :: MatrixXf weights;             ///< array-matrix of weights

protected:

  std :: deque < Eigen :: ArrayXf > history; ///< deque for the convergency monitoring
  Eigen :: VectorXf theta;                    ///< array of means

  std :: function < float(const float &) > activation; ///< pointer to activation function
  std :: function < float(const float &) > gradient;   ///< pointer to gradient function

  int32_t batch;                  ///< batch size
  int32_t outputs;                ///< number of hidden units
  int32_t epochs_for_convergency; ///< number of stable epochs requested for the convergency

  float convergency_atol;     ///< Absolute tolerance requested for the convergency
  float decay;                ///< Weight decay scale factor

  static float precision;     ///< Parameter that controls numerical precision of the weight updates.

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
  * @param decay Weight decay scale factor.
  *
  */
  BasePlasticity (const int32_t & outputs, const int32_t & batch_size,
    int32_t activation=transfer_t :: linear,
    update_args optimizer=update_args(optimizer_t :: sgd),
    weights_initialization weights_init=weights_initialization(weights_init_t :: normal),
    int32_t epochs_for_convergency=1, float convergency_atol=0.01f,
    float decay=0.f);


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
  * The signature of the function is totally equivalent to the the
  * Python counterpart except by the pointer arrays which require the
  * dimension size as extra parameters.
  *
  * @note This function must be called before the predict member-function.
  * A check is performed internally to ensure it.
  *
  * @param X array in ravel format of the input variables/features
  * @param n_samples dimension of the X matrix, i.e. the number of rows
  * @param n_features dimension of the X matrix, i.e. the number of cols
  * @param num_epochs Number of epochs for model convergency.
  * @param seed Random seed number for the batch subdivisions.
  * @param callback Callback function to call at each batch evaluation.
  *
  * @tparam Callback void lambda function which can use member variables.
  *
  */
  template < class Callback = std :: function < void (BasePlasticity *) > >
  void fit (float * X, const int & n_samples, const int & n_features, const int & num_epochs,
    int seed=42, Callback callback=[](BasePlasticity *) -> void {});

  /**
  * @brief Train the model/encoder
  *
  * @details Proxy function for a user interface compatible with Eigen matrix.
  *
  * @param X Eigen matrix of the input variables/features.
  * @param num_epochs Number of epochs for model convergency.
  * @param seed Random seed number for the batch subdivisions.
  * @param callback Callback function to call at each batch evaluation.
  *
  * @tparam Callback void lambda function which can use member variables.
  *
  */
  template < class Callback = std :: function < void (BasePlasticity *) > >
  void fit (const Eigen :: MatrixXf & X, const int & num_epochs,
    int32_t seed=42, Callback callback=[](BasePlasticity *) -> void {});

  /**
  * @brief Predict the model/encoder
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
  float * predict (float * X, const int32_t & n_samples, const int32_t & n_features);

  /**
  * @brief Predict the model/encoder
  *
  * @details Proxy function for a user interface compatible with Eigen matrix.
  *
  * @param X Eigen matrix of the input variables/features.
  *
  * @return The array of encoded features.
  *
  */
  float * predict (const Eigen :: MatrixXf & X);

  /**
  * @brief Save the current weight matrix.
  *
  * @details The weights matrix is saved in binary format.
  * The first value of the file is an integer corresponding to the number of
  * weights (rows x cols) of the weight matrix, followed by the (float)
  * weight matrix in ravel format.
  *
  * @param filename Filename or path where the file is saved.
  *
  */
  void save_weights (const std :: string & filename);

  /**
  * @brief Load the current weight matrix.
  *
  * @details The weights matrix is loaded according to the format
  * specified in the save_weights function, i.e. the first first value of
  * the file is an integer corresponding to the number of weights
  * (rows x cols) of the weight matrix, followed by the (float) weight
  * matrix in ravel format.
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
  * @param X Batch of data.
  * @param output Output of the model as computed by the _predict function
  *
  * @return Matrix of updates (aka dW) for weights.
  *
  */
  virtual Eigen :: MatrixXf weights_update (const Eigen :: MatrixXf & X, const Eigen :: MatrixXf & output) = 0;

  /**
  * @brief Check the input dimensions.
  *
  * @note The function checks if the given dimensions are consistent with
  * the input ones.
  *
  * @param n_features dimension of the X matrix, i.e. the number of cols
  *
  */
  void check_dims (const int32_t & n_features);

  /**
  * @brief Check if the model is already fitted.
  *
  * @note The function checks if function fit has been already called
  * before the prediction.
  * The check is performed on the value of the output array
  *
  */
  void check_is_fitted ();

  /**
  * @brief Check the given parameters.
  *
  * @note The function checks if the input variable epochs_for_convergency
  * is positive defined and greater than 1
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
  * @param vec Vector containing updates to check for the convergence estimation
  *
  */
  bool check_convergence (const Eigen :: ArrayXf & vec);

  /**
  * @brief Core function of the fit formula
  *
  * @note This is the core function of the fit procedure, i.e the
  * function in which the computation of the training step is performed
  *
  * @param X Eigen matrix of the input variables/features.
  * @param num_epochs Number of epochs for model convergency.
  * @param seed Random seed number for the batch subdivisions.
  * @param callback Callback function to call at each batch evaluation.
  *
  * @tparam Callback void lambda function which can use member variables.
  *
  */
  template < class Callback >
  void _fit (const Eigen :: MatrixXf & X, const int32_t & num_epochs,
    const int32_t & seed, Callback callback);

  /**
  * @brief Core function of the predict formula
  *
  * @note This abstract function implements the predict rule of
  * the model given the data matrix.
  *
  * @param data Input matrix of data
  *
  * @return Output matrix of the model.
  *
  */
  virtual Eigen :: MatrixXf _predict (const Eigen :: MatrixXf & data);

};


#endif // __base_h__
