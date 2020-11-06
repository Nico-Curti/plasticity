#ifdef HOPFIELD

#include <hopfield.h>

Hopfield :: Hopfield (const int & outputs, const int & batch_size,
                      float mu, float sigma, float epsilon, float delta, float p, int k, int seed
                      ) : BasePlasticity (outputs, batch_size, transfer :: _linear_, mu, sigma, epsilon, seed),
                          k (k), delta (delta), p (p)
{
  this->fire_indices.reset(new int[this->outputs]);
  this->delta_indices.reset(new int[this->outputs]);
  this->yl.reset(new float[this->outputs * this->batch]);

  this->check_params();
}


Hopfield :: Hopfield (const Hopfield & b) : BasePlasticity (b)
{
  this->fire_indices.reset(new int[b.outputs]);
  this->delta_indices.reset(new int[b.outputs]);

  this->k = b.k;
  this->delta = b.delta;
  this->p = b.p;
}

Hopfield & Hopfield :: operator = (const Hopfield & b)
{
  BasePlasticity :: operator = (b);

  this->fire_indices.reset(new int[b.outputs]);
  this->delta_indices.reset(new int[b.outputs]);

  this->k = b.k;
  this->delta = b.delta;
  this->p = b.p;

  return *this;
}


void Hopfield :: check_params ()
{
  if ( this->k < 2 )
  {
    std :: cerr << "k must be an integer bigger or equal than 2" << std :: endl;
    std :: exit(ERROR_K_POSITIVE);
  }
}

void Hopfield :: weights_update (float * X, const int & n_features, float * weights_update)
{

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < outputs; ++i)
  {
    const int idx = i * this->batch;
    fire_indices[i]  = idx + this->batch - 1;
    delta_indices[i] = idx + this->batch - this->k;

    std :: fill_n(this->yl.get() + idx, this->yl.get() + idx + this->batch, 0.f);
  }

#ifdef _OPENMP
  #pragma omp single
  {
#endif

    std :: sort(this->fire_indices.get(), this->fire_indices.get() + this->outputs,
                [&](const int & i, const int & j)
                {
                  return this->output[i] < this->output[j];
                });

    std :: sort(this->delta_indices.get(), this->delta_indices.get() + this->outputs,
                [&](const int & i, const int & j)
                {
                  return this->output[i] < this->output[j];
                });

#ifdef _OPENMP
  } // end single section
#endif

  static float nc;

  nc = 0.f;

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->outputs; ++i)
  {
    const int up_idx = fire_indices[i];
    const int rest_idx = delta_indices[i];
    this->yl[up_idx] = 1.
    this->yl[rest_idx] = -this->delta;

    this->theta[i] += this->yl[up_idx] * this->output[up_idx];
    this->theta[i] += this->yl[rest_idx] * this->output[rest_idx];
  }

  // MISS gemm_nn  yl @ X - theta * self.weights
  // weights (outputs, n_features)
  // this->nweights = this->outputs * n_features;
  //

#ifdef _OPENMP
  #pragma omp for reduction (max : nc)
#endif
  for (int i = 0; i < this->nweights; ++i)
  {
    // TODO
    const float sum = std :: inner_product(yl + ..., yl + ..., X + ..., 0.f);
    const float w_up = sum - theta * this->weights[i];
    const float abs_w_up = std :: fabs(w_up);

    nc = nc < abs_w_up ? abs_w_up : nc;
    weights_update[i] = w_up;
  }

  nc = 1.f / std :: max(nc, BasePlasticity :: precision);


#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
    weights_update[i] *= nc;

}


void Hopfield :: normalize_weights ()
{
  if ( this->p != 2 )
  {

    const int p = this->p;

#ifdef _OPENMP

    #pragma omp for
    for (int i = 0; i < this->nweights; ++i)
    {
      wi = this->weights[i];
      this->weights[i] = std :: sign(wi) * std :: pow(std :: fabs(wi), p - 1);
    }

#else

    std :: transform(this->weights.get(), this->weights.get() + this->nweights,
                     this->weights.get(),
                     [&](const float & wi)
                     {
                       return std :: sign(wi) * std :: pow(std :: fabs(wi), p - 1);
                     });

#endif

  }
}

#endif