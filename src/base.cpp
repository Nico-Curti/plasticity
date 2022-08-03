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

#include <base.h>

float BasePlasticity :: precision = 1e-30f;

BasePlasticity :: BasePlasticity () : optimizer (), w_init (), weights (),
  history (), theta (), activation (nullptr), gradient (nullptr),
  batch (100), outputs (100), epochs_for_convergency (0), convergency_atol (0.f),
  decay (0.f)
{
}

BasePlasticity :: BasePlasticity (const int32_t & outputs, const int32_t & batch_size,
  int32_t activation,
  update_args optimizer,
  weights_initialization weights_init,
  int32_t epochs_for_convergency, float convergency_atol,
  float decay
  ) : optimizer (optimizer), w_init (weights_init), weights (), history (),
      theta (), activation (nullptr), gradient (nullptr),
      batch (batch_size), outputs (outputs), epochs_for_convergency (epochs_for_convergency),
      convergency_atol (convergency_atol), decay (decay)
{
  // correct epochs_for_convergency
  //this->epochs_for_convergency = std :: max(this->epochs_for_convergency, 1);

  this->activation = transfer :: activate( activation );
  this->gradient   = transfer :: gradient( activation );
}

BasePlasticity :: BasePlasticity (const BasePlasticity & b)
{
  this->activation = b.activation;
  this->gradient   = b.gradient;

  this->batch    = b.batch;
  this->outputs  = b.outputs;
  this->epochs_for_convergency = b.epochs_for_convergency;

  this->convergency_atol = b.convergency_atol;

  this->optimizer = b.optimizer;
  this->w_init = b.w_init;

  this->weights = b.weights;

  //this->theta = b.theta; // it is useless
}

BasePlasticity & BasePlasticity :: operator = (const BasePlasticity & b)
{
  this->activation = b.activation;
  this->gradient   = b.gradient;

  this->batch    = b.batch;
  this->outputs  = b.outputs;
  this->epochs_for_convergency = b.epochs_for_convergency;

  this->convergency_atol = b.convergency_atol;

  this->optimizer = b.optimizer;
  this->w_init = b.w_init;

  this->weights = b.weights;

  //this->theta = b.theta; // it is useless

  return *this;
}

float * BasePlasticity :: predict (float * X, const int32_t & n_samples, const int32_t & n_features)
{
  // convert the input array to Eigen matrix
  Eigen :: Map < Eigen :: Matrix < float, Eigen :: Dynamic, Eigen :: Dynamic, Eigen :: RowMajor > > data(X, n_samples, n_features);
  // call the "real" function
  return this->predict(data);
}

float * BasePlasticity :: predict (const Eigen :: MatrixXf & X)
{
  // extracthe the number of features as the number of columns of the input matrix
  const int32_t n_features = X.cols();

  // check if the model has already stored the weights matrix (aka the fit function has already run)
  this->check_is_fitted ();
  // check if the input dimensions are consistent with the training ones
  this->check_dims (n_features);

  // perform the prediction using the core (overrided) function
  Eigen :: MatrixXf output = this->_predict (X);

  return output.data();
}

void BasePlasticity :: save_weights (const std :: string & filename)
{
  // check if the model has already stored the weights matrix (aka the fit function has already run)
  this->check_is_fitted ();

  // open the output stream file as binary
  std :: ofstream os(filename, std :: ios :: out | std :: ios :: binary | std :: ios :: trunc);

  // temporary variables for the matrix shape
  typename Eigen :: MatrixXf :: Index rows = this->weights.rows();
  typename Eigen :: MatrixXf :: Index cols = this->weights.cols();

  // dump the shape variables
  os.write((char *) (&rows), sizeof ( typename Eigen :: MatrixXf :: Index ) );
  os.write((char *) (&cols), sizeof ( typename Eigen :: MatrixXf :: Index ) );
  // dump the matrix data buffer
  os.write((char *) this->weights.data(), rows * cols * sizeof ( typename Eigen :: MatrixXf :: Scalar) );
  // close the file stream
  os.close();
}

void BasePlasticity :: load_weights (const std :: string & filename)
{
  // check if the provided file exists trying to open it
  if ( ! utils :: file_exists(filename) )
    // throw the exception with the appropriated error
    throw std :: runtime_error("File not found. Given : " + filename);

  // open the file stream as binary
  std :: ifstream is(filename, std :: ios :: in | std :: ios :: binary);

  // temporary variables for the matrix shape
  typename Eigen :: MatrixXf :: Index rows = 0;
  typename Eigen :: MatrixXf :: Index cols = 0;

  // read the shape variables
  is.read((char*) (&rows), sizeof ( typename Eigen :: MatrixXf :: Index ) );
  is.read((char*) (&cols), sizeof ( typename Eigen :: MatrixXf :: Index ) );
  // resize the weights matrix
  this->weights.resize(rows, cols);
  // read the matrix data buffer
  is.read( (char *) this->weights.data(), rows * cols * sizeof ( typename Eigen :: MatrixXf :: Scalar) );
  // close the file stream
  is.close();
}

float * BasePlasticity :: get_weights ()
{
  // extract the pointer to the data stored into the weight Eigen Matrix
  return this->weights.data();
}


// Private members

void BasePlasticity :: check_dims (const int32_t & n_features)
{
  // Check the shape consistency between the input data (n_samples, n_features)
  // and the weights matrix (outputs, n_features)
  // This function is used just to be sure that the dataset provided for the
  // prediction are consistent with the data (shape) provided for the training
  if ( this->outputs * n_features != this->weights.size() )
    throw std :: runtime_error("Invalid dimensions found. The input (n_samples, n_features)"
                               "shape is inconsistent with the number of weights (" +
                                std :: to_string(this->weights.size()) + ")");
}

void BasePlasticity :: check_is_fitted ()
{
  // If the model has not yet called the fit function the weights matrix is empty!
  if ( this->weights.rows() == 0 && this->weights.cols() == 0 )
    throw std :: runtime_error("Fitted error. The model is not fitted yet.\n"
                               "Please call the fit function before using the predict member.");
}

void BasePlasticity :: check_params ()
{
  // the number of epochs for the convergency must be a positive non null integer!
  if ( this->epochs_for_convergency <= 0 )
    throw std :: runtime_error("epochs_for_convergency must be an integer bigger or equal than 1");
}

bool BasePlasticity :: check_convergence (const Eigen :: ArrayXf & vec)
{
  // If the history queue is not full append the last (current) vector to the history
  if ( static_cast < int32_t >(this->history.size()) < this->epochs_for_convergency )
  {
    this->history.emplace_back(vec);
    return false;
  }

  // Otherwise evaluate the distances between the current vector
  // and the history arrays. The distance is evaluated as the abs differences
  // between each historical vector and the current one.
  // If the maximum value of the differences is lower than the given tollerance
  // the convergency is reached and a stop is returned.

  for (int32_t i = 0; i < this->epochs_for_convergency; ++i)
  {
    // compute the vector of abs differences
    const bool equal = vec.isApprox(this->history[i], this->convergency_atol);

    if ( !equal )
      return true;
  }

  // Since this is the case in which the history queue is full
  // we have to remove the older vector and append the current one (LIFO behavior)
  this->history.pop_front();
  this->history.emplace_back(vec);

  return false;
}

Eigen :: MatrixXf BasePlasticity :: _predict (__unused const Eigen :: MatrixXf & data)
{
  return Eigen :: MatrixXf {};
}

