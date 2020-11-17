#ifndef __hopfield_h__
#define __hopfield_h__

#include <base.h>

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

  std :: unique_ptr < float[] > yl;          ///< matrix of updates
  std :: unique_ptr < int[] > fire_indices;  ///< array of indices related to the maximum output

  int k; ///< ranking parameter

  float delta; ///< Strength of the anti-hebbian learning
  float p;     ///< Lebesque norm of weights


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
  * @param optimizer update_args Optimizer object.
  * @param mu Mean of the gaussian distribution that initializes the weights.
  * @param sigma Standard deviation of the gaussian distribution that initializes the weights.
  * @param epochs_for_convergency Number of stable epochs requested for the convergency.
  * @param convergency_atol Absolute tolerance requested for the convergency.
  * @param delta Strength of the anti-hebbian learning
  * @param p Lebesgue norm of the weights.
  * @param k Ranking parameter, must be integer that is bigger or equal than 2.
  * @param seed Random number generator seed.
  *
  */
  Hopfield (const int & outputs, const int & batch_size,
            update_args optimizer=update_args(optimizer_t :: _sgd),
            float mu=0.f, float sigma=1.f,
            int epochs_for_convergency=1, float convergency_atol=1e-2f,
            float delta=.4f, float p=2.f,
            int k=2, int seed=42);

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
  * @param n_features dimension of the X matrix, i.e. the number of cols.
  * @param weights_update Array/matrix of updates for weights.
  *
  */
  void weights_update (float * X, const int & n_features, float * weights_update);

  /**
  * @brief Apply the Lebesgue norm to the weights.
  *
  * @note The function implements the Lebesque norm on the weight matrix following the equation:
  *
  * \code{.py}
  * W = sign(W) * abs(W)**(p - 1)
  * \endcode
  *
  */
  void normalize_weights ();

};


#endif // __hopfield_h__
