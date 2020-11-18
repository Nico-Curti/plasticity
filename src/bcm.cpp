#include <bcm.h>

BCM :: BCM (const int & outputs, const int & batch_size,
            int activation, update_args optimizer, weights_initialization weights_init,
            int epochs_for_convergency, float convergency_atol,
            float interaction_strength
            ) : BasePlasticity (outputs, batch_size, activation, optimizer, weights_init, epochs_for_convergency, convergency_atol)
{
  this->init_interaction_matrix(interaction_strength);
}


BCM :: BCM (const BCM & b) : BasePlasticity (b)
{
}

BCM & BCM :: operator = (const BCM & b)
{
  BasePlasticity :: operator = (b);

  return *this;
}


void BCM :: init_interaction_matrix (const float & interaction_strength)
{
  this->interaction_matrix.reset(new float[this->outputs * this->outputs]);

  if (interaction_strength != 0.f)
  {
    for (int i = 0; i < this->outputs; ++i)
      for (int j = 0; j < this->outputs; ++j)
      {
        const int idx = i * this->outputs + j;
        this->interaction_matrix[idx] = i == j ? 1.f : -interaction_strength;
      }

    // map the matrix to the eigen format
    Eigen :: Map < Eigen :: Matrix < float, Eigen :: Dynamic, Eigen :: Dynamic, Eigen :: RowMajor > > L(interaction_matrix.get(), outputs, outputs);
    // compute the inverse of the matrix
    auto inverse = L.inverse();
    // re-map the result into the member variable
    Eigen :: Map < Eigen :: MatrixXf > ( this->interaction_matrix.get(), inverse.rows(), inverse.cols() ) = inverse;
  }

  else
  {
    for (int i = 0; i < this->outputs; ++i)
      for (int j = 0; j < this->outputs; ++j)
      {
        const int idx = i * this->outputs + j;
        this->interaction_matrix[idx] = i == j ? 1.f : 0.f;
      }
  }
}


void BCM :: weights_update (float * X, const int & n_features, float * weights_update)
{
  static float nc;

  nc = 0.f;

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
    weights_update[i] = 0.f;

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->outputs; ++i)
  {
    const int idx = i * this->batch;
    this->theta[i] = std :: accumulate(this->output.get() + idx,
                                       this->output.get() + idx + this->batch,
                                       0.f,
                                       [](const float & res, const float & xi)
                                       {
                                        return res + xi * xi;
                                       }) / this->batch;
  }

  // interaction_matrix (outputs, outputs)
  // output (outputs, batch)
  // X (batch, n_features)
  // theta (outputs)

#ifdef _OPENMP
  #pragma omp for collapse (2)
#endif
  for (int i = 0; i < this->outputs; ++i)
    for (int j = 0; j < this->batch; ++j)
    {

      float interaction_sum = 0.f;

      for (int k = 0; k < this->outputs; ++k)
      {
        const int idx = i * this->outputs + k;
        const float interaction = this->interaction_matrix[idx];
        const float out_old = this->output[k * this->batch + j];
        const float phi = out_old * (out_old - this->theta[k]);

        interaction_sum += interaction * phi;
      }

      const int idx = i * this->batch + j;
      const float out_old = this->output[idx];
      const float out_new = this->gradient(this->activation(out_old));

      float * xi = X + j * n_features;
      float * wi = weights_update + i * n_features;

      for (int k = 0; k < n_features; ++k)
        wi[k] += interaction_sum * out_new * xi[k];

      this->output[idx] = out_new;
    }

#ifdef _OPENMP
  #pragma omp for reduction (max : nc)
#endif
  for (int i = 0; i < this->nweights; ++i)
  {
    const float out = std :: fabs(weights_update[i]);
    nc = nc < out ? out : nc;
  }

#ifdef _OPENMP
  #pragma omp single
#endif
  nc = 1.f / std :: max(nc, BasePlasticity :: precision);

#ifdef _OPENMP
  #pragma omp for
#endif
  for (int i = 0; i < this->nweights; ++i)
    //weights_update[i] *= nc;
    weights_update[i] *= - nc; // Add the minus for compatibility with optimization algorithms
}

