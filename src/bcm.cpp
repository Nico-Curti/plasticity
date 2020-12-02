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

  // output (outputs, batch)
  // X (batch, n_features)
  // theta (outputs)

#ifdef _OPENMP
  #pragma omp for collapse (2)
#endif
  for (int i = 0; i < this->outputs; ++i)
    for (int j = 0; j < this->batch; ++j)
    {
      const int idx = i * this->batch + j;
      const float out = this->output[idx];
      // Use the Law and Cooper update rule (1994)
      const float phi = out * (out - this->theta[i]) / (this->theta[i] + BasePlasticity :: precision);

      float * xi = X + j * n_features;
      float * wi = weights_update + i * n_features;

      for (int k = 0; k < n_features; ++k)
        wi[k] += phi * xi[k];
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



void BCM :: _predict (const float * A, const float * B, float * C, const int & N, const int & M, const int & K)
{

  // A = weights (outputs x n_features) (M x K)
  // B = X (batch x n_features) (N x K)
  // C = output (outputs x batch) (M x N)
  // N = batch
  // M = outputs
  // K = n_features

  // interaction_matrix (outputs, outputs)

  // TODO: miss interaction matrix in the evaluation

#ifdef __avx__

const int prev_end = (K % 8 == 0) ? (K - 8) : (K >> 3) << 3;

#endif

  // weights @ X.T
  // weights (outputs x n_features)
  // X (batch x n_features)
  // out (outputs x batch)

#ifdef _OPENMP
#pragma omp for collapse (2)
#endif
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
    {
      const int index = i * N + j;
      const int idx1 = i * K;
      const int idx2 = j * K;

  #ifdef __avx__ // TO CHECK

      float sum = 0.f;

      for (int k = 0; k < prev_end; k += 8)
      {
        __m256 a256 = _mm256_load_ps(A + idx1 + k);
        __m256 b256 = _mm256_load_ps(B + idx2 + k);
        __m256 c256 = _mm256_dp_ps(a256, b256, 0xff);
        sum += ((float*)&c256)[0];
      }

      sum += std :: inner_product(A + idx1 + prev_end, A + idx1 + prev_end + K,
                                  B + idx2 + prev_end, 0.f);
  #else

      const float sum  = std :: inner_product(A + idx1, A + idx1 + K,
                                              B + idx2, 0.f);

  #endif // __avx__

      C[index] = this->activation(sum);
    }
}
