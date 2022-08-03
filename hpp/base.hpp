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

#ifndef __base_hpp__
#define __base_hpp__

#include <base.h>

template < class Callback >
void BasePlasticity :: fit (float * X, const int32_t & n_samples, const int32_t & n_features,
  const int32_t & num_epochs, int32_t seed, Callback callback)
{
  // convert the input array to Eigen matrix
  // NOTE: perform a copy of the array to avoid possible troubles
  // in relation to the index permutation (?)
  Eigen :: Map < Eigen :: Matrix < float, Eigen :: Dynamic, Eigen :: Dynamic, Eigen :: RowMajor > > data(X, n_samples, n_features);

  // call the "real" function
  this->fit (data, num_epochs, seed, callback);
}

template < class Callback >
void BasePlasticity :: fit (const Eigen :: MatrixXf & X, const int32_t & num_epochs,
  int32_t seed, Callback callback)
{
  if (this->batch > X.rows())
    throw std :: runtime_error("Incorrect batch_size found. "
      "The batch_size must be less or equal to the number of samples. "
      "Given " + std :: to_string(this->batch) + " for " +
      std :: to_string(X.rows()) + " samples");

  // extract the number of features as the number of columns of the input matrix
  const int32_t n_features = X.cols();

  // allocate the weights matrix
  this->weights = Eigen :: MatrixXf(this->outputs, n_features);
  // init the weight matrix using the given initializer
  this->w_init.init(this->weights.data(), this->outputs, n_features);

  // init the optimizer object with the required parameters
  this->optimizer.init_arrays(this->weights.rows(), this->weights.cols());

  // call the core fit function
  this->_fit (X, num_epochs, seed, callback);
}

template < class Callback >
void BasePlasticity :: _fit (const Eigen :: MatrixXf & X, const int32_t & num_epochs,
  const int32_t & seed, Callback callback)
{
  // compute the number of possible batches
  const int32_t num_batches = X.rows() / this->batch;
  // extract the number of matrix shape
  const int32_t n_samples = X.rows();
  const int32_t n_features = X.cols();

#if EIGEN_VERSION_AT_LEAST(3, 3, 90)
  // Build the index permutation generator
  std :: vector < int32_t > batch_indices(n_samples);
  std :: iota(batch_indices.begin(), batch_indices.end(), 0);
#else
  // The solution with eigen permutation is very very very slow...
  Eigen :: PermutationMatrix < Eigen :: Dynamic, Eigen :: Dynamic > permutation (n_samples);
  // init with the identity mat
  permutation.setIdentity();
#endif

  // init theta as zeros array
  this->theta = Eigen :: VectorXf :: Zero(this->outputs);

  // init the random number generator for the permutation
  std :: mt19937 engine(seed);

  // initialize the (possible) parallel environment
  Eigen :: initParallel();

  // start the loop along the epochs
  for (int32_t epoch = 0; epoch < num_epochs; ++epoch)
  {

    // set the initial accumulator to zeros
    // This vector will be check for the estimation
    // of model convergence at each epoch
    Eigen :: ArrayXf sum_theta = Eigen :: ArrayXf :: Zero(this->outputs);

#if EIGEN_VERSION_AT_LEAST(3, 3, 90)
    // Perform an index permutation at each epoch
    std :: shuffle(batch_indices.begin(), batch_indices.end(), engine);
    // apply the index permutation on the data
    Eigen :: MatrixXf X_perm = X(batch_indices, Eigen :: all); // permute rows
#else
    std :: shuffle(permutation.indices().data(),
                   permutation.indices().data() + permutation.indices().size(),
                   engine);
    // apply the index permutation on the data
    Eigen :: MatrixXf X_perm = permutation * X; // permute rows
#endif

#ifdef __verbose__

    std :: cout << RESET_COUT << "Epoch " << epoch + 1 << "/" << num_epochs << std :: endl;
    auto timer  = utils :: what_time_is_it_now();

#endif // __verbose__

    // start the evaluation of the batches
    for (int32_t i = 0; i < num_batches; ++i)
    {

      // Get the batch data as block starting from the i*batch row
      // and the first (0) column with a shape given by (batch_size, num_features)
      auto batch_data = X_perm.block(i * this->batch, 0, this->batch, n_features);

      // perform the prediction of the model with the current weight matrix

      Eigen :: MatrixXf output = this->_predict(batch_data);

      // compute the gradient of the weights matrix (aka dW)
      Eigen :: MatrixXf weights_update = this->weights_update(batch_data, output);

      // (eventually) perform a weight decay
      if (this->decay != 0.f)
        weights_update -= this->decay * this->weights;

      // perform the update of the weights using the properly set optimizer
      this->optimizer.update(epoch + 1, this->weights, weights_update);

      // update the convergence vector
      sum_theta += this->theta.array();

#ifdef __verbose__

      // print the progress bar of the training
      utils :: print_progress (i + 1, num_batches, timer);

#endif // __verbose__

      callback(this);

    } // end for batches

#ifdef __verbose__

    std :: cout << std :: endl;

#endif // __verbose__

    // check if the model has reached the convergency
    if ( this->check_convergence(sum_theta * (1.f / num_batches)) )
    {

#ifdef __verbose__

      std :: cout << "Early stopping: the training has reached the convergency criteria" << std :: endl;

#endif // __verbose__

      break;
    }


  } // end for epoch
}


#endif // __base_hpp__
