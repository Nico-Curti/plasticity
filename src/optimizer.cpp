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

#include <optimizer.h>

float update_args :: epsil = 1e-6f;

update_args :: update_args () : m (), v (), type (-1),
  learning_rate (0.f), momentum (0.f),
  decay (0.f), B1 (0.f), B2 (0.f), rho (0.f)
{
}

update_args :: update_args (const int32_t & type,
  float learning_rate, float momentum,
  float decay, float B1, float B2, float rho) : m (), v (), type (type),
                                                learning_rate (learning_rate), momentum (momentum),
                                                decay (decay), B1 (B1), B2 (B2), rho (rho)
{
#ifdef DEBUG

  assert (type >= optimizer_t :: adam && type <= optimizer_t :: sgd);

#endif
}

void update_args :: init_arrays (const int32_t & rows, const int32_t & cols)
{
  this->m = Eigen :: MatrixXf :: Zero(rows, cols);
  this->v = Eigen :: MatrixXf :: Zero(rows, cols);
}

update_args & update_args :: operator = (const update_args & args)
{
  this->type = args.type;

  this->learning_rate = args.learning_rate;
  this->momentum = args.momentum;
  this->decay = args.decay;
  this->B1 = args.B1;
  this->B2 = args.B2;
  this->rho = args.rho;

  this->m = args.m;

  return *this;
}

update_args :: update_args (const update_args & args) : type (args.type), learning_rate (args.learning_rate),
                                                        momentum (args.momentum), decay (args.decay),
                                                        B1 (args.B1), B2 (args.B2), rho (args.rho)
{
  this->m = args.m;
  this->v = args.v;
}

void update_args :: update ( const int32_t & iteration, Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update )
{

  if ( this->m.size() != weights.size() )
    throw std :: runtime_error("Invalid number of weights found. Given " + std :: to_string(weights.size()) + ". Aspected " + std :: to_string(this->m.size()));

  switch ( this->type )
  {
    case optimizer_t :: adam:              adam_update(iteration, weights, weights_update);
    break;
    case optimizer_t :: momentum:          momentum_update(weights, weights_update);
    break;
    case optimizer_t :: nesterov_momentum: nesterov_momentum_update(weights, weights_update);
    break;
    case optimizer_t :: adagrad:           adagrad_update(weights, weights_update);
    break;
    case optimizer_t :: rmsprop:           rmsprop_update(weights, weights_update);
    break;
    case optimizer_t :: adadelta:          adadelta_update(weights, weights_update);
    break;
    case optimizer_t :: adamax:            adamax_update(iteration, weights, weights_update);
    break;
    case optimizer_t :: sgd:               sgd_update(weights, weights_update);
    break;

  }

  this->learning_rate *= 1.f / (this->decay * iteration + 1.f);
  this->learning_rate  = this->learning_rate < 0.f ? 0.f : this->learning_rate;
}

void update_args :: adam_update (const int32_t & iteration, Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update)
{
  const float a_t = this->learning_rate * math :: sqrt(1.f - math :: pow(this->B2, iteration)) / (1.f - math :: pow(this->B1, iteration));

  this->m = this->m * this->B1 + (1.f - this->B1) * weights_update;
  this->v = this->v * this->B2 + (1.f - this->B2) * weights_update.cwiseProduct(weights_update);

  weights = weights.array() - a_t * this->m.array() / (this->v.cwiseSqrt().array() + update_args :: epsil);
}


void update_args :: sgd_update (Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update)
{
  weights = weights - this->learning_rate * weights_update;
}

void update_args :: momentum_update (Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update)
{
  this->v = this->momentum * this->v - this->learning_rate * weights_update;
  weights = weights + this->v;
}

void update_args :: nesterov_momentum_update (Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update)
{
  this->v = this->momentum * this->v - this->learning_rate * weights_update;
  weights = weights + this->momentum * this->v - this->learning_rate * weights_update;
}

void update_args :: adagrad_update (Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update)
{
  this->v = weights_update.cwiseProduct(weights_update);
  weights = weights.array() - this->learning_rate * weights_update.array() / (this->v.cwiseSqrt().array() + update_args :: epsil);
}

void update_args :: rmsprop_update (Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update)
{
  this->v = this->rho * this->v + (1.f - this->rho) * weights_update.cwiseProduct(weights_update);
  weights = weights.array() - this->learning_rate * weights_update.array() / (this->v.cwiseSqrt().array() + update_args :: epsil);
}

void update_args :: adadelta_update (Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update)
{
  this->v = this->rho * this->v + (1.f - this->rho) * weights_update.cwiseProduct(weights_update);
  Eigen :: MatrixXf update = weights_update.array() * ( this->m.cwiseSqrt().array() + update_args :: epsil) / (this->v.cwiseSqrt().array() + update_args :: epsil);
  weights = weights - this->learning_rate * update;
  this->m = this->rho * this->m + (1.f - this->rho) * update.cwiseProduct(update);
}

void update_args :: adamax_update (const int32_t & iteration, Eigen :: MatrixXf & weights, const Eigen :: MatrixXf & weights_update)
{
  const float a_t = this->learning_rate / (1.f - math :: pow(this->B1, iteration));

  this->m = this->m * this->B1 + (1.f - this->B1) * weights_update;
  this->v = weights_update.cwiseAbs().cwiseMax(this->B2 * this->v);
  weights = weights.array() - a_t * this->m.array() / (this->v.array() + update_args :: epsil);
}
