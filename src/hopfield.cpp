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

#include <hopfield.h>

Hopfield :: Hopfield (const int32_t & outputs, const int32_t & batch_size,
  update_args optimizer,
  weights_initialization weights_init,
  int32_t epochs_for_convergency, float convergency_atol,
  float decay,
  float delta, float p, int32_t k
  ) : BasePlasticity (outputs, batch_size, transfer_t :: linear,
                      optimizer, weights_init, epochs_for_convergency,
                      convergency_atol, decay),
      k (k), delta (delta), p (p)
{
  this->check_params();
}


Hopfield :: Hopfield (const Hopfield & b) : BasePlasticity (b), k (b.k), delta (b.delta), p (b.p)
{
}

Hopfield & Hopfield :: operator = (const Hopfield & b)
{
  BasePlasticity :: operator = (b);

  this->k = b.k;
  this->delta = b.delta;
  this->p = b.p;

  return *this;
}


void Hopfield :: check_params ()
{
  // The value of the K variable must be positive and greater than 2
  if ( this->k < 2 )
    throw std :: runtime_error("k must be an integer bigger or equal than 2");
}


Eigen :: MatrixXf Hopfield :: weights_update (const Eigen :: MatrixXf & X, const Eigen :: MatrixXf & output)
{
  // compute the columns re-order for the Krotov approximation

  // init the interaction matrix with a zero matrix
  Eigen :: MatrixXf yl = Eigen :: MatrixXf :: Zero(this->outputs, this->batch);
  int32_t col_index = 0;

  // sort the output columns
  // NOTE: the following for loop can be rewritten as for (auto && col : output.colwise())
  //       using directly the col vector but it is available only with Eigen version > 3.3.90.
  //       In the same way we can rewrite the sorting algorithm with the simplified .begin(), .end()
  //       member functions.
  //       We use the older version of the Eigen syntax just to improve the retro-compatibility of the
  //       library
  for (int32_t i = 0; i < output.cols(); ++i)
  {
    auto col = output.col(i);
    Eigen :: VectorXi order = Eigen :: VectorXi :: LinSpaced(this->outputs, 0, this->outputs);
    std :: sort(order.data(), order.data() + this->outputs,
                [&](const int32_t & xi, const int32_t & xj)
                {
                  return col[xi] < col[xj];
                });
    const int32_t row_index = order[this->outputs - 1];
    yl(row_index, col_index) = 1.f;

    const int32_t decrease_index = order[this->outputs - this->k];
    yl(decrease_index, col_index) = -this->delta;

    ++ col_index;
  }

  this->theta = yl.cwiseProduct(output).rowwise().sum();

  // compute the weights updates using the Hopfield formulation
  Eigen :: MatrixXf weights_update = (yl * X).array() - this->weights.array().colwise() * this->theta.array();

  // normalize the weights update by the maximum value
  // to avoid numerical instabilities
  const float max_abs_val = 1.f / weights_update.cwiseAbs().maxCoeff();
  // Add the minus for compatibility with optimization algorithms
  weights_update = weights_update.array() * (-max_abs_val);

  return weights_update;
}


Eigen :: MatrixXf Hopfield :: _predict (const Eigen :: MatrixXf & data)
{
  // apply Lebesgue norm
  Eigen :: MatrixXf wnorm = (this->p != 2.f) ? this->weights.unaryExpr(
    [&](const float & w) -> float {
      return std :: copysign(std :: pow(std :: fabs(w), this->p - 1.f), w);
    }) : this->weights;
  // Compute the output as W @ X
  Eigen :: MatrixXf output = wnorm * data.transpose();

  return output;
}
