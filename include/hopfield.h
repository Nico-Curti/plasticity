#ifdef HOPFIELD

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

  std :: unique_ptr < float[] > yl; ///< matrix of updates
  std :: unique_ptr < int[] > fire_indices;  ///< array of indices related to the maximum output
  std :: unique_ptr < int[] > delta_indices; ///< array-of indices related to the k-th maximum output

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
  * @param mu Mean of the gaussian distribution that initializes the weights.
  * @param sigma Standard deviation of the gaussian distribution that initializes the weights.
  * @param epsilon Starting learning rate.
  * @param delta Strength of the anti-hebbian learning
  * @param p Lebesgue norm of the weights.
  * @param k Ranking parameter, must be integer that is bigger or equal than 2.
  * @param seed Random number generator seed.
  *
  */
  Hopfield (const int & outputs, const int & batch_size,
            float mu=0.f, float sigma=1.f, float epsilon=2e-2f, float delta=.4f, float p=2.f,
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
  * We override the base method to pre-allocate the arrays required by the current
  * specialization (in particular the array yl).
  *
  * @param X array in ravel format of the input variables/features
  * @param n_samples dimension of the X matrix, i.e. the number of rows
  * @param n_features dimension of the X matrix, i.e. the number of cols
  * @param num_epochs Number of epochs for model convergency.
  *
  */
  void fit (float * X, const int & n_samples, const int & n_features, const int & num_epochs);


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

#endif
