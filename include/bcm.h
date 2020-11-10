#ifndef __bcm_h__
#define __bcm_h__

#include <base.h>
#include <Eigen/Dense>

/**
* @class BCM
*
* @brief Bienenstock, Cooper and Munro algorithm (BCM).
*
* @details The idea of BCM theory is that for a random sequence of input patterns a synapse
* is learning to differentiate between those stimuli that excite the postsynaptic
* neuron strongly and those stimuli that excite that neuron weakly.
* Learned BCM feature detectors cannot, however, be simply used as the lowest layer
* of a feedforward network so that the entire network is competitive to a network of
* the same size trained with backpropagation algorithm end-to-end.
*
*/
class BCM : public BasePlasticity
{

  std :: unique_ptr < float[] > interaction_matrix; ///< interaction matrix between weights

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
  * @param activation Index of the activation function.
  * @param optimizer update_args Optimizer object.
  * @param mu Mean of the gaussian distribution that initializes the weights.
  * @param sigma Standard deviation of the gaussian distribution that initializes the weights.
  * @param interaction_strength Set the lateral interaction strength between weights.
  * @param seed Random number generator seed.
  *
  */
  BCM (const int & outputs, const int & batch_size, int activation=transfer :: _logistic_,
       update_args optimizer=update_args(optimizer_t :: _sgd),
       float mu=0.f, float sigma=1.f, float interaction_strength=0.f, int seed=42);

  // Copy Operator and Copy Constructor

  /**
  * @brief Copy constructor.
  *
  * @details The copy constructor provides a deep copy of the object, i.e. all the
  * arrays are copied and not moved.
  *
  * @param b BCM object
  *
  */
  BCM (const BCM & b);

  /**
  * @brief Copy operator.
  *
  * @details The operator performs a deep copy of the object and if there are buffers
  * already allocated, the operatore deletes them and then re-allocates an appropriated
  * portion of memory.
  *
  * @param b BCM object
  *
  */
  BCM & operator = (const BCM & b);

  // Destructor

  /**
  * @brief Destructor.
  *
  * @details Completely delete the object and release the memory of the arrays.
  *
  */
  ~BCM () = default;


private:

  /**
  * @brief Compute the weights update using the BCM learning rule.
  *
  * @note
  *
  * @param X array in ravel format of the input variables/features.
  * @param n_features dimension of the X matrix, i.e. the number of cols.
  * @param weights_update Array/matrix of updates for weights.
  *
  */
  void weights_update (float * X, const int & n_features, float * weights_update);

  /**
  * @brief Initialize the weights interaction matrix.
  *
  * @note This function is the only one which requires the Eigen library support.
  * The Eigen library is used in the evaluation of the inverse matrix.
  *
  * @param interaction_strenght Set the lateral interaction strenght between weights.
  *
  */
  void init_interaction_matrix (const float & interaction_strenght);

};


#endif // __bcm_h__
