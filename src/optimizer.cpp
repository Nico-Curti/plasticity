#include <optimizer.h>

float update_args :: epsil = 1e-6f;

update_args :: update_args () : m (nullptr), v (nullptr), nweights (0), type (-1), learning_rate (0.f), momentum (0.f), decay (0.f), B1 (0.f), B2 (0.f), rho (0.f), l2norm (false), clip (false)
{
}

update_args :: update_args (const int & type, float learning_rate, float momentum, float decay, float B1, float B2, float rho, bool l2norm, bool clip)
                           : m (nullptr), v (nullptr), nweights (0), type (type), learning_rate (learning_rate), momentum (momentum), decay (decay), B1 (B1), B2 (B2), rho (rho), l2norm (l2norm), clip (clip)
{
#ifdef DEBUG

  assert (type >= _adam && type <= _sgd);

#endif

}

void update_args :: init_arrays (const int & nweights)
{
  this->nweights = nweights;

  m.reset(new float[nweights]);
  v.reset(new float[nweights]);

  std :: fill_n (m.get(), nweights, 0.f);
  std :: fill_n (v.get(), nweights, 0.f);
}


update_args & update_args :: operator = (const update_args & args)
{
  nweights = args.nweights;
  type = args.type;

  learning_rate = args.learning_rate;
  momentum = args.momentum;
  decay = args.decay;
  B1 = args.B1;
  B2 = args.B2;
  rho = args.rho;
  l2norm = args.l2norm;
  clip = args.clip;

  m.reset(new float[nweights]);
  v.reset(new float[nweights]);

  std :: copy_n (args.m.get(), nweights, m.get());
  std :: copy_n (args.v.get(), nweights, v.get());

  return *this;
}

update_args :: update_args (const update_args & args) : nweights (args.nweights), type (args.type), learning_rate (args.learning_rate), momentum (args.momentum), decay (args.decay), B1 (args.B1), B2 (args.B2), rho (args.rho), l2norm (args.l2norm), clip (args.clip)
{
  m.reset(new float[nweights]);
  v.reset(new float[nweights]);

  std :: copy_n (args.m.get(), nweights, m.get());
  std :: copy_n (args.v.get(), nweights, v.get());

}

void update_args :: update ( const int & iteration, float * weights, float * weights_update, const int & nweights )
{

  if ( this->nweights != nweights )
  {
    std :: cerr << "Invalid number of weights found. Given " << nweights << ". Aspected " << this->nweights << std :: endl;
    throw ERROR_NWEIGHTS;
  }

  if ( this->l2norm )
    this->norm_value(weights_update, nweights);

  if ( this->clip )
    this->clip_value(weights_update, nweights);

  switch ( this->type )
  {
    case optimizer_t :: _adam:              adam_update(iteration, weights, weights_update);
    break;
    case optimizer_t :: _momentum:          momentum_update(weights, weights_update);
    break;
    case optimizer_t :: _nesterov_momentum: nesterov_momentum_update(weights, weights_update);
    break;
    case optimizer_t :: _adagrad:           adagrad_update(weights, weights_update);
    break;
    case optimizer_t :: _rmsprop:           rmsprop_update(weights, weights_update);
    break;
    case optimizer_t :: _adadelta:          adadelta_update(weights, weights_update);
    break;
    case optimizer_t :: _adamax:            adamax_update(iteration, weights, weights_update);
    break;
    case optimizer_t :: _sgd:               sgd_update(weights, weights_update);
    break;

  }
  this->learning_rate *= 1.f / (this->decay * iteration + 1.f);
  this->learning_rate  = this->learning_rate < 0.f ? 0.f : this->learning_rate;
}


void update_args :: adam_update (const int & iteration, float * weights, float * weights_update)
{
  const float a_t = this->learning_rate * math :: sqrt(1.f - math :: pow(this->B2, iteration)) / (1.f - math :: pow(this->B1, iteration));

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
  {
    this->m[i] = this->m[i] * this->B1 + (1.f - this->B1) * weights_update[i];
    this->v[i] = this->v[i] * this->B2 + (1.f - this->B2) * weights_update[i] * weights_update[i];

    weights[i] -= a_t * this->m[i] / (math :: sqrt( this->v[i] ) + update_args :: epsil );
  }
}



void update_args :: sgd_update (float * weights, float * weights_update)
{
#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
    weights[i] -= this->learning_rate * weights_update[i];
}


void update_args :: momentum_update (float * weights, float * weights_update)
{
#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
  {
    this->v[i]  = this->momentum * this->v[i] - this->learning_rate * weights_update[i];
    weights[i] += this->v[i];
  }
}


void update_args :: nesterov_momentum_update (float * weights, float * weights_update)
{
#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
  {
    this->v[i]  = this->momentum * this->v[i] - this->learning_rate * weights_update[i];
    weights[i] += ( this->momentum * this->v[i] - this->learning_rate * weights_update[i]);
  }
}



void update_args :: adagrad_update (float * weights, float * weights_update)
{
#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
  {
    this->v[i] += weights_update[i] * weights_update[i];
    weights[i] -= this->learning_rate * weights_update[i] / (math :: sqrt(this->v[i]) + update_args :: epsil);
  }
}


void update_args :: rmsprop_update (float * weights, float * weights_update)
{
#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
  {
    this->v[i]  = this->rho * this->v[i] + (1.f - this->rho) * weights_update[i] * weights_update[i];
    weights[i] -= this->learning_rate * weights_update[i] / (math :: sqrt(this->v[i]) + update_args :: epsil);
  }
}


void update_args :: adadelta_update (float * weights, float * weights_update)
{

  float update = 0.f;

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
  {
    this->v[i]  = this->rho * this->v[i] + (1.f - this->rho) * weights_update[i] * weights_update[i];
    update      = weights_update[i] * ( math :: sqrt(this->m[i]) + update_args :: epsil) / (math :: sqrt(this->v[i]) + update_args :: epsil);
    weights[i] -= this->learning_rate * update;
    this->m[i]  = this->rho * this->m[i] + (1.f - this->rho) * update * update;
  }
}



void update_args :: adamax_update (const int & iteration, float * weights, float * weights_update)
{
  const float a_t = this->learning_rate / (1.f - math :: pow(this->B1, iteration));

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
  {
    this->m[i] = this->m[i] * this->B1 + (1.f - this->B1) * weights_update[i];
    this->v[i] = std :: max(this->B2 * this->v[i], std :: fabs(weights_update[i]));

    weights[i] -= a_t * this->m[i] / (this->v[i] + update_args :: epsil);
  }

}

void update_args :: norm_value (float * arr, const int & size)
{

#ifdef _OPENMP

  float norm = 0;

  #pragma omp for
  for (int i = 0; i < size; ++i)
    norm += arr[i] * arr[i];

  norm = math :: rsqrt(norm);

  #pragma omp for
  for (int i = 0; i < size; ++i)
    arr[i] *= norm;

#else

  const float norm = math :: rsqrt(std :: inner_product(arr, arr + size, arr, 0.f));

  std :: transform(arr, arr + size, arr,
                   [&](const float & x)
                   {
                     return x * norm;
                   });

#endif

}

void update_args :: clip_value (float * arr, const int & size)
{

#ifdef _OPENMP

  #pragma omp for
  for (int i = 0; i < size; ++i)
    arr[i] = std :: clamp(arr[i], -1.f, 1.f);

#else

  std :: transform(arr, arr + size, arr,
                   [&](const float & x)
                   {
                     return std :: clamp(x, -1.f, 1.f);
                   });

#endif

}
