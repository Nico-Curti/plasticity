#ifdef HOPFIELD

#include <hopfield.h>

Hopfield :: Hopfield (const int & outputs, const int & batch_size,
                      float mu, float sigma, float epsilon, float delta, float p, int k, int seed
                      ) : BasePlasticity (outputs, batch_size, transfer :: _linear_, mu, sigma, epsilon, seed),
                          k (k), delta (delta), p (p)
{
  this->fire_indices.reset(new int[this->batch]);
  this->delta_indices.reset(new int[this->batch]);

  std :: iota(fire_indices.get(), fire_indices.get() + this->batch, 0);
  std :: iota(delta_indices.get(), delta_indices.get() + this->batch, 0);

  this->check_params();
}


Hopfield :: Hopfield (const Hopfield & b) : BasePlasticity (b)
{
  this->fire_indices.reset(new int[b.batch]);
  std :: copy_n (b.fire_indices.get(), b.batch, this->fire_indices.get());
  this->delta_indices.reset(new int[b.batch]);
  std :: copy_n (b.delta_indices.get(), b.batch, this->delta_indices.get());

  this->k = b.k;
  this->delta = b.delta;
  this->p = b.p;
}

Hopfield & Hopfield :: operator = (const Hopfield & b)
{
  BasePlasticity :: operator = (b);

  this->fire_indices.reset(new int[b.batch]);
  std :: copy_n (b.fire_indices.get(), b.batch, this->fire_indices.get());
  this->delta_indices.reset(new int[b.batch]);
  std :: copy_n (b.delta_indices.get(), b.batch, this->delta_indices.get());

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

void Hopfield :: fit (float * X, const int & n_samples, const int & n_features, const int & num_epochs)
{
  this->nweights = this->outputs * n_features;
  this->yl.reset(new float[this->nweights]);

  BasePlasticity :: fit(X, n_samples, n_features, num_epochs);
}

void Hopfield :: weights_update (float * X, const int & n_features, float * weights_update)
{

#ifdef _OPENMP
  #pragma omp single
  {
#endif

    int idx = -1;

    std :: sort(this->fire_indices.get(), this->fire_indices.get() + this->batch,
                [&](const int & i, const int & j)
                {
                  idx = (++ idx) * this->outputs + this->batch - 1;
                  return this->output[idx] > this->output[idx + this->outputs];
                });

    idx = -1;
    std :: sort(this->delta_indices.get(), this->delta_indices.get() + this->batch,
                [&](const int & i, const int & j)
                {
                  idx = (++ idx) * this->outputs + this->batch - this->k;
                  return this->output[idx] > this->output[idx + this->outputs];
                });

#ifdef _OPENMP
  } // end single section
#endif

  // TODO: theta is an array of

  static float nc;

  nc = 0.f;

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->batch; ++i)
  {
    const int up_idx = fire_indices[i] * this->batch + i;
    const int rest_idx = delta_indices[i] * this->batch + i;
    yl[up_idx] = 1.
    yl[rest_idx] = -this->delta;

    theta += yl[up_idx] * this->output[up_idx];
    theta += yl[rest_idx] * this->output[rest_idx];
  }

  // MISS gemm_nn  yl @ X - theta * self.weights
  // weights (outputs, n_features)

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