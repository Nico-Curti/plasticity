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

#ifndef __update_args_h__
#define __update_args_h__

#include <fmath.h>       // fast math functions

#include <iostream>      // std :: cerr
#include <unordered_map> // std :: unordered_map
#include <Eigen/Dense>   // Eigen classes


enum optimizer_t { adam = 0, momentum, nesterov_momentum,
                   adagrad, rmsprop, adadelta,
                   adamax, sgd
}; ///< optimizer types

namespace optimizer
{

static const std :: unordered_map < std :: string, int32_t > get_optimizer {
    {"adam"              , adam},
    {"momentum"          , momentum},
    {"nesterov_momentum" , nesterov_momentum},
    {"adagrad"           , adagrad},
    {"rmsprop"           , rmsprop},
    {"adadelta"          , adadelta},
    {"adamax"            , adamax},
    {"sgd"               , sgd},
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

  Eigen :: MatrixXf m; ///< Adam supporting array
  Eigen :: MatrixXf v; ///< Adam supporting array

public:

  int32_t type;        ///< Optimization type to use

  float learning_rate; ///< Learning rate value
  float momentum;      ///< Momentum parameter
  float decay;         ///< Decay parameter
  float B1;            ///< Adam-like parameter
  float B2;            ///< Adam-like parameter
  float rho;           ///< Decay factor

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
  *
  */
  update_args (const int32_t & type, float learning_rate=0.02, float momentum=.9f,
    float decay=0.0001, float B1=.9f, float B2=.999f, float rho=0.f);

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
  * @param rows Number of weights/parameters rows to update.
  * @param cols Number of weights/parameters cols to update.
  */
  void init_arrays (const int32_t & rows, const int32_t & cols);

  /**
  * @brief Update the given parameters using the optimization algorithm
  *
  * @details This is the core functio of the object.
  *
  * @param iteration Current iteration number
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void update ( const int32_t & iteration, Eigen :: MatrixXf & weights,
    const Eigen :: MatrixXf & weights_update );

private:

  /**
  * @brief Adam optimization step
  *
  * @param iteration Current iteration number
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void adam_update ( const int32_t & iteration, Eigen :: MatrixXf & weights,
    const Eigen :: MatrixXf & weights_update );

  /**
  * @brief Stochastic Gradient Descent optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void sgd_update ( Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update );

  /**
  * @brief Momentum optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void momentum_update ( Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update );

  /**
  * @brief Nesterov momentum optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void nesterov_momentum_update ( Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update );

  /**
  * @brief AdaDrad optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void adagrad_update ( Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update );

  /**
  * @brief RMSProp optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void rmsprop_update ( Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update );

  /**
  * @brief AdaDelta optimization step
  *
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void adadelta_update ( Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update );

  /**
  * @brief AdaMax optimization step
  *
  * @param iteration Current iteration number
  * @param weights Array of input parameters
  * @param weights_update Array of input gradients.
  *
  */
  void adamax_update ( const int32_t & iteration, Eigen :: MatrixXf & weights,
    const Eigen :: MatrixXf & weights_update );

};

#endif // __update_args_h__
