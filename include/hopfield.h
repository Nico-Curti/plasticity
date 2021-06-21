#ifndef __hopfield_h__
#define __hopfield_h__

#include <base.hpp>

/**
* @class Hopfield
*
* @brief
*
* @details
*
*/
class Hopfield : public BasePlasticity
{

  int k;        ///< ranking parameter
  float delta;  ///< Strength of the anti-hebbian learning
  float p;      ///< Lebesque norm of weights


public:


  // Constructor

  /**
  * @brief Construct the object using the list of training parameters.
  *
  * @details The constructor follows the same nomenclature of the Python counterpart.
  *
  * @note
  *
  * @param outputs Number of hidden units.
  * @param batch_size Size of the minibatch.
  * @param optimizer update_args Optimizer object (default=SGD algorithm).
  * @param weights_init weights_initialization object (default=uniform initialization in [-1, 1]).
  * @param epochs_for_convergency Number of stable epochs requested for the convergency.
  * @param convergency_atol Absolute tolerance requested for the convergency.
  * @param decay Weight decay scale factor.
  * @param delta Strength of the anti-hebbian learning
  * @param p Lebesgue norm of the weights.
  * @param k Ranking parameter, must be integer that is bigger or equal than 2.
  *
  */
  Hopfield (const int & outputs, const int & batch_size,
            update_args optimizer=update_args(optimizer_t :: sgd),
            weights_initialization weights_init=weights_initialization(weights_init_t :: normal),
            int epochs_for_convergency=1, float convergency_atol=0.01,
            float decay=0.f,
            float delta=.4f, float p=2.f,
            int k=2);

  // Copy Operator and Copy Constructor

  /**
  * @brief Copy constructor.
  *
  * @details The copy constructor provides a deep copy of the object, i.e. all the
  * arrays are copied and not moved.
  *
  * @param b Hopfield object
  *
  */
  Hopfield (const Hopfield & b);

  /**
  * @brief Copy operator.
  *
  * @details The operator performs a deep copy of the object and if there are buffers
  * already allocated, the operatore deletes them and then re-allocates an appropriated
  * portion of memory.
  *
  * @param b Hopfield object
  *
  */
  Hopfield & operator = (const Hopfield & b);

  // Destructor

  /**
  * @brief Destructor.
  *
  * @details Completely delete the object and release the memory of the arrays.
  *
  */
  ~Hopfield () = default;

private:

  /**
  * @brief Check the given parameters.
  *
  * @note The function checks if the input variable k is positive defined and greater than 2
  *
  */
  void check_params ();

  /**
  * @brief Approximation introduced by Krotov.
  *
  * @note Instead of solving dynamical equations we use the currents as a proxy
  * for ranking of the final activities.
  *
  * @param X array in ravel format of the input variables/features.
  * @param output Output of the model as computed by the _predict function
  *
  */
  Eigen :: MatrixXf weights_update (const Eigen :: MatrixXf & X, const Eigen :: MatrixXf & output);

  /**
  * @brief Apply the Lebesgue norm to the weights.
  *
  * @note The function implements the Lebesgue norm on the weight matrix following the equation:
  *
  * \f[
  * W = sign(W) * abs(W)**(p - 1)
  * \f]
  *
  */
  void normalize_weights ();

  /**
  * @brief Core function of the predict formula
  *
  * @note The function computes the output as W @ X.T.
  * We use the GEMM algorithm with OpenMP support for a fast evaluation
  *
  * @param data Input matrix of data
  *
  * @return Output matrix of the model.
  *
  */
  Eigen :: MatrixXf _predict (const Eigen :: MatrixXf & data);

};


#endif // __hopfield_h__
