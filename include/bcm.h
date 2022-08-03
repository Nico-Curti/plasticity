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

#ifndef __bcm_h__
#define __bcm_h__

#include <base.hpp>

/**
* @class BCM
*
* @brief Bienenstock, Cooper and Munro algorithm (BCM).
*
* @details The idea of BCM theory is that for a random sequence of input
* patterns a synapse is learning to differentiate between those stimuli
* that excite the postsynaptic neuron strongly and those stimuli that
* excite that neuron weakly.
* Learned BCM feature detectors cannot, however, be simply used as the
* lowest layer of a feedforward network so that the entire network is
* competitive to a network of the same size trained with backpropagation
* algorithm end-to-end.
*
*/
class BCM : public BasePlasticity
{

  Eigen :: MatrixXf interaction_matrix; ///< interaction matrix between weights
  float memory_factor; ///< Memory factor for weighting the theta updates.

public:


  // Constructor

  /**
  * @brief Construct the object using the list of training parameters.
  *
  * @details The constructor follows the same nomenclature of the Python counterpart.
  *
  * @param outputs Number of hidden units.
  * @param batch_size Size of the minibatch.
  * @param activation Index of the activation function.
  * @param optimizer update_args Optimizer object (default=SGD algorithm).
  * @param weights_init weights_initialization object (default=uniform initialization in [-1, 1]).
  * @param epochs_for_convergency Number of stable epochs requested for the convergency.
  * @param convergency_atol Absolute tolerance requested for the convergency.
  * @param decay Weight decay scale factor.
  * @param memory_factor Memory factor for weighting the theta updates.
  * @param interaction_strength Set the lateral interaction strength between weights.
  *
  */
  BCM (const int32_t & outputs, const int32_t & batch_size,
    int32_t activation=transfer_t :: logistic,
    update_args optimizer=update_args(optimizer_t :: sgd),
    weights_initialization weights_init=weights_initialization(weights_init_t :: normal),
    int32_t epochs_for_convergency=1, float convergency_atol=0.01f,
    float decay=0.f, float memory_factor=0.5f,
    float interaction_strength=0.f);

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
  * @note Perform the updating rule of the BCM algorithm provided by
  * the model of Law and Cooper(1994), i.e
  *
  * \f[
  * \frac{dw_i}{dt} = y (y - \theta_M) x_i / \theta_M
  * \theta_M = E[y^2]
  * \f]
  *
  * The Law and Cooper form has all of the same fixed points as the Intrator and Cooper
  * form, but speed of synaptic modification increases when the threshold is small,
  * and decreases as the threshold increases.
  * The practical result is that the simulation can be run with artificially high
  * learning rates, and wild oscillations are reduced. This form has been used primarily
  * when running simulations of networks, where the run-time of the simulation can
  * be prohibitive.
  *
  * @param X Batch of data.
  * @param output Output of the model as computed by the _predict function
  *
  * @return weights_update Matrix of updates (aka dW) for weights.
  *
  */
  Eigen :: MatrixXf weights_update (const Eigen :: MatrixXf & X, const Eigen :: MatrixXf & output);

  /**
  * @brief Initialize the weights interaction matrix.
  *
  * @note Evaluate the interaction matrix between neurons. The default model doesn't
  * provide lateral interactions between neurons, i.e the interaction matrix is the
  * identity matrix. The case of negative interaction strenght corrensponds to an
  * inhibition of the neurons, while a positive interaction strenght corresponds to
  * a promotion.
  * The interaction matrix is computed as
  * \f[
  * L = I - interaction
  * \f]
  * i.e a square matrix (outputs, outputs) with all the elements along the diagonal
  * equal to 1 and all the other entries as -interaction_strenght.
  * Since in the predict function the inverse of this matrix is required, the
  * inverse computation is performed before the storing into this function.
  *
  * @param interaction_strenght Set the lateral interaction strenght between weights.
  *
  */
  void init_interaction_matrix (const float & interaction_strenght);

  /**
  * @brief Core function of the predict formula
  *
  * @note The function computes the output
  * \f[
  * y = \sigma(\sum_i w_i x_i)
  * \f]
  * If the lateral interactions are set the output function becomes
  * \f[
  * y = \sigma(\sum_i L_i^{-1} w_i x_i)
  * \f]
  * where \f$L\f$ is the interaction matrix between the neurons.
  *
  * @param data Input matrix of data.
  *
  * @return Output matrix of the model.
  *
  */
  Eigen :: MatrixXf _predict (const Eigen :: MatrixXf & data);


};


#endif // __bcm_h__
