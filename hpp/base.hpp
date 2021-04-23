#ifndef __base_hpp__
#define __base_hpp__

#include <base.h>

template < class Callback >
void BasePlasticity :: fit (float * X, const int & n_samples, const int & n_features, const int & num_epochs, int seed, Callback callback)
{
  // convert the input array to Eigen matrix
  // NOTE: perform a copy of the array to avoid possible troubles
  // in relation to the index permutation (?)
  Eigen :: Map < Eigen :: Matrix < float, Eigen :: Dynamic, Eigen :: Dynamic, Eigen :: RowMajor > > data(X, n_samples, n_features);

  // call the "real" function
  this->fit (data, num_epochs, seed, callback);
}

template < class Callback >
void BasePlasticity :: fit (const Eigen :: MatrixXf & X, const int & num_epochs, int seed, Callback callback)
{
  // extracthe the number of features as the number of columns of the input matrix
  const int n_features = X.cols();

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
void BasePlasticity :: _fit (const Eigen :: MatrixXf & X, const int & num_epochs, const int & seed, Callback callback)
{
  // compute the number of possible batches
  const int num_batches = X.rows() / this->batch;
  // extract the number of matrix shape
  const int n_samples = X.rows();
  const int n_features = X.cols();

#if EIGEN_VERSION_AT_LEAST(3, 3, 90)
  // Build the index permutation generator
  std :: vector < int > batch_indices(n_samples);
  std :: iota(batch_indices.begin(), batch_indices.end(), 0);
#else
  // The solution with eigen permutation is very very very slow...
  Eigen :: PermutationMatrix < Eigen :: Dynamic, Eigen :: Dynamic > permutation (n_samples);
  // init with the identity mat
  permutation.setIdentity();
#endif

  // init the random number generator for the permutation
  std :: mt19937 engine(seed);

  // start the loop along the epochs
  for (int epoch = 0; epoch < num_epochs; ++epoch)
  {


#if EIGEN_VERSION_AT_LEAST(3, 3, 90)
    // Perform an index permutation at each epoch
    std :: shuffle(batch_indices.begin(), batch_indices.end(), engine);
    // apply the index permutation on the data
    auto X_perm = X(batch_indices, Eigen :: all); // permute rows
#else
    std :: shuffle(permutation.indices().data(), permutation.indices().data() + permutation.indices().size(), engine);
    // apply the index permutation on the data
    auto X_perm = permutation * X; // permute rows
#endif

#ifdef __verbose__

    std :: cout << RESET_COUT << "Epoch " << epoch + 1 << "/" << num_epochs << std :: endl;
    auto timer  = utils :: what_time_is_it_now();

#endif // __verbose__

    // start the evaluation of the batches
    for (int i = 0; i < num_batches; ++i)
    {

      // Get the batch data as block starting from the i*batch row and the first (0) column
      // with a shape given by (batch_size, num_features)
      auto batch_data = X_perm.block(i * this->batch, 0, this->batch, n_features);

      // (eventually) perform the weights normalization/standardization
      this->normalize_weights();

      // perform the prediction of the model with the current weight matrix

      Eigen :: MatrixXf output = this->_predict(batch_data);

      // compute the gradient of the weights matrix (aka dW)
      Eigen :: MatrixXf weights_update = this->weights_update(batch_data, output);

      // perform the update of the weights using the properly set optimizer
      this->optimizer.update(epoch + 1, this->weights, weights_update);

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
    if ( this->check_convergency() )
    {

#ifdef __verbose__

      std :: cout << "Early stopping: the training has reached the convergency criteria" << std :: endl;

#endif // __verbose__

      break;
    }


  } // end for epoch
}


#endif // __base_hpp__
