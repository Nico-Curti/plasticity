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

#include <weights.h>

weights_initialization :: weights_initialization () : type (-1), mu (0.f), sigma (0.f), scale (0.f)
{
}

weights_initialization :: weights_initialization (const int32_t & type,
  float mu, float sigma, float scale, int32_t seed) : type (type), mu (mu), sigma (sigma), scale (scale)
{
  this->engine = std :: mt19937(seed);

  // check the initialization type (greater than zero and included into the get map of available values)
  if ( type < 0 || type >= static_cast < int32_t >(weights_init :: get_weights.size()) )
    throw std :: runtime_error("Invalid initialization function");
}

weights_initialization & weights_initialization :: operator = (const weights_initialization & args)
{
  engine = args.engine;
  type = args.type;
  mu = args.mu;
  sigma = args.sigma;
  scale = args.scale;

  return *this;
}

weights_initialization :: weights_initialization (const weights_initialization & args) :
  engine (args.engine), type (args.type), mu (args.mu), sigma (args.sigma), scale (args.scale)
{
}

void weights_initialization :: init (float * weights, const int32_t & inputs, const int32_t & outputs)
{
  switch (this->type)
  {
    case weights_init_t :: zeros :          return this->zeros(weights, inputs, outputs);
    case weights_init_t :: ones :           return this->ones(weights, inputs, outputs);
    case weights_init_t :: uniform :        return this->uniform(weights, inputs, outputs);
    case weights_init_t :: normal :         return this->normal(weights, inputs, outputs);
    case weights_init_t :: lecun_uniform :  return this->lecun_uniform(weights, inputs, outputs);
    case weights_init_t :: glorot_uniform : return this->glorot_uniform(weights, inputs, outputs);
    case weights_init_t :: lecun_normal :   return this->lecun_normal(weights, inputs, outputs);
    case weights_init_t :: glorot_normal :  return this->glorot_normal(weights, inputs, outputs);
    case weights_init_t :: he_uniform :     return this->he_uniform(weights, inputs, outputs);
    case weights_init_t :: he_normal :      return this->he_normal(weights, inputs, outputs);
    default:
    {
      throw std :: runtime_error("Invalid initialization function");
    } break;
  }
}

// Private members

void weights_initialization :: zeros (float * weights, const int32_t & inputs, const int32_t & outputs)
{
  const int32_t size = inputs * outputs;

  std :: fill_n(weights, size, 0.f);
}

void weights_initialization :: ones (float * weights, const int32_t & inputs, const int32_t & outputs)
{
  const int32_t size = inputs * outputs;

  std :: fill_n(weights, size, 1.f);
}

void weights_initialization :: uniform (float * weights, const int32_t & inputs, const int32_t & outputs)
{
  const int32_t size = inputs * outputs;
  std :: uniform_real_distribution < float > random_uniform (-this->scale, this->scale);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_uniform(this->engine);
                     });
}

void weights_initialization :: normal (float * weights, const int32_t & inputs, const int32_t & outputs)
{
  const int32_t size = inputs * outputs;
  std :: normal_distribution < float > random_normal (this->mu, this->sigma);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_normal(this->engine);
                     });
}

void weights_initialization :: lecun_uniform (float * weights, const int32_t & inputs, const int32_t & outputs)
{
  const int32_t size = inputs * outputs;
  const float scale = math :: sqrt(3.f / inputs);
  std :: uniform_real_distribution < float > random_uniform (-scale, scale);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_uniform(this->engine);
                     });
}

void weights_initialization :: glorot_uniform (float * weights, const int32_t & inputs, const int32_t & outputs)
{
  const int32_t size = inputs * outputs;
  const float scale = math :: sqrt(6.f / (inputs + outputs));
  std :: uniform_real_distribution < float > random_uniform (-scale, scale);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_uniform(this->engine);
                     });
}

void weights_initialization :: lecun_normal (float * weights, const int32_t & inputs, const int32_t & outputs)
{
  const int32_t size = inputs * outputs;
  const float sigma = math :: sqrt(1.f / inputs);
  std :: normal_distribution < float > random_normal (this->mu, sigma);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_normal(this->engine);
                     });
}

void weights_initialization :: glorot_normal (float * weights, const int32_t & inputs, const int32_t & outputs)
{
  const int32_t size = inputs * outputs;
  const float sigma = math :: sqrt(2.f / (inputs + outputs));
  std :: normal_distribution < float > random_normal (this->mu, sigma);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_normal(this->engine);
                     });
}

void weights_initialization :: he_uniform (float * weights, const int32_t & inputs, const int32_t & outputs)
{
  const int32_t size = inputs * outputs;
  const float scale = math :: sqrt(6.f / inputs);
  std :: uniform_real_distribution < float > random_uniform (-scale, scale);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_uniform(this->engine);
                     });
}

void weights_initialization :: he_normal (float * weights, const int32_t & inputs, const int32_t & outputs)
{
  const int32_t size = inputs * outputs;
  const float sigma = math :: sqrt(2.f / inputs);
  std :: normal_distribution < float > random_normal (this->mu, sigma);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_normal(this->engine);
                     });
}
