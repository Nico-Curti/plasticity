#include <base.h>

std :: mt19937 BasePlasticity :: engine = std :: mt19937(0);
float BasePlasticity :: precision = 1e-30f;

BasePlasticity :: BasePlasticity () : optimizer (), output (nullptr), weights (nullptr), activation (nullptr), gradient (nullptr),
                                      batch (100), outputs (100), nweights (0), mu (0.f), sigma (1.f)
{
}

BasePlasticity :: BasePlasticity (const int & outputs, const int & batch_size, int activation,
                                  update_args optimizer,
                                  float mu, float sigma, int seed
                                  ) : optimizer (optimizer), output (nullptr), weights (nullptr), activation (nullptr), gradient (nullptr),
                                      batch (batch_size), outputs (outputs), nweights (0), mu (mu), sigma (sigma)
{
  this->activation = transfer :: activate( activation );
  this->gradient   = transfer :: gradient( activation );

  this->theta.reset(new float[this->outputs]);

  BasePlasticity :: engine = std :: mt19937(seed);
}

BasePlasticity :: BasePlasticity (const BasePlasticity & b)
{
  this->nweights   = b.nweights;
  this->activation = b.activation;
  this->gradient   = b.gradient;

  this->batch    = b.batch;
  this->outputs  = b.outputs;
  this->nweights = b.nweights;

  this->mu      = b.mu;
  this->sigma   = b.sigma;

  this->optimizer = b.optimizer;

  this->weights.reset(new float[b.nweights]);
  std :: copy_n (b.weights.get(), b.nweights, this->weights.get());

  this->theta.reset(new float[b.outputs]);
  //std :: copy_n (b.theta.get(), b.outputs, this->theta.get()); // it is useless
}

BasePlasticity & BasePlasticity :: operator = (const BasePlasticity & b)
{
  this->nweights   = b.nweights;
  this->activation = b.activation;
  this->gradient   = b.gradient;

  this->batch    = b.batch;
  this->outputs  = b.outputs;
  this->nweights = b.nweights;

  this->mu      = b.mu;
  this->sigma   = b.sigma;

  this->optimizer = b.optimizer;

  this->weights.reset(new float[b.nweights]);
  std :: copy_n (b.weights.get(), b.nweights, this->weights.get());

  this->theta.reset(new float[b.outputs]);
  //std :: copy_n (b.theta.get(), b.outputs, this->theta.get()); // it is useless

  return *this;
}


void BasePlasticity :: fit (float * X, const int & n_samples, const int & n_features, const int & num_epochs)
{
  this->nweights = this->outputs * n_features;
  this->weights.reset(new float[this->nweights]);

  const int outputs = this->outputs * this->batch;
  this->output.reset(new float[outputs]);

  std :: normal_distribution < float > random_normal (this->mu, this->sigma);

  std :: generate_n (this->weights.get(), this->nweights,
                     [&]()
                     {
                       return random_normal(BasePlasticity :: engine);
                     });

  this->optimizer.init_arrays(this->nweights);

  this->_fit (X, num_epochs, n_features, n_samples);
}

float * BasePlasticity :: predict (const float * X, const int & n_samples, const int & n_features)
{
  this->check_is_fitted ();
  this->check_dims (n_features);

  const int outputs = this->outputs * n_samples;
  this->output.reset(new float[outputs]);

#ifdef _OPENMP

  #pragma omp parallel
  {

#endif

    this->_predict (this->weights.get(), X, this->output.get(), n_samples, this->outputs, n_features);

#ifdef _OPENMP

  } // end parallel section

#endif

  return this->output.get();
}

void BasePlasticity :: save_weights (const std :: string & filename)
{
  this->check_is_fitted ();

  std :: ofstream os(filename, std :: ios :: out | std :: ios :: binary);

  os.write( (const char *) &this->nweights, sizeof( int ));
  os.write( reinterpret_cast < char* >(this->weights.get()), sizeof(float) * this->nweights);

  os.close();
}

void BasePlasticity :: load_weights (const std :: string & filename)
{

  if ( ! utils :: file_exists(filename) )
  {
    std :: cerr << "File not found. Given : " << filename << std :: endl;
    throw 1;
  }

  std :: ifstream is(filename, std :: ios :: in | std :: ios :: binary);

  is.read(reinterpret_cast < char* >(&this->nweights), sizeof(int));

  this->weights.reset(new float[this->nweights]);
  is.read(reinterpret_cast < char* >(this->weights.get()), sizeof(float) * this->nweights);

  is.close();
}

float * BasePlasticity :: get_weights ()
{
  return this->weights.get();
}


// Private members

void BasePlasticity :: check_dims (const int & n_features)
{
  if ( this->outputs * n_features != this->nweights )
  {
    std :: cerr << "Invalid dimensions found. The input (n_samples, n_features) shape is inconsistent with the number of weights (" << this->nweights << ")" << std :: endl;
    throw ERROR_DIMENSIONS;
  }
}

void BasePlasticity :: check_is_fitted ()
{
  if ( ! this->weights )
  {
    std :: cerr << "Fitted error. The model is not fitted yet." << std :: endl
                << "Please call the fit function before using the predict member." << std :: endl;
    throw ERROR_FITTED;
  }
}

void BasePlasticity :: normalize_weights ()
{
}


void BasePlasticity :: _fit (float * X, const int & num_epochs, const int & n_features, const int & n_samples)
{
  const int num_batches = n_samples / this->batch;

  std :: unique_ptr < float[] > weights_update (new float[this->nweights]);
  std :: unique_ptr < int[] > batch_indices(new int [num_batches]);
  std :: iota(batch_indices.get(), batch_indices.get() + num_batches, 0);

  for (int epoch = 0; epoch < num_epochs; ++epoch)
  {
    std :: shuffle(batch_indices.get(), batch_indices.get() + num_batches, BasePlasticity :: engine);

    std :: cout << RESET_COUT << "Epoch " << epoch << "/" << num_epochs << std :: endl;
    auto timer  = utils :: what_time_is_it_now();

    for (int i = 0; i < num_batches; ++i)
    {

#ifdef _OPENMP

      #pragma omp parallel
      {

#endif
        float * batch_data = X + i * n_features * this->batch;

        this->normalize_weights();
        this->_predict(this->weights.get(), batch_data, this->output.get(), this->batch, this->outputs, n_features);
        this->weights_update(batch_data, n_features, weights_update.get());

        // update weights

        this->optimizer.update(this->weights.get(), weights_update.get(), this->nweights);

#ifdef _OPENMP

      } // end parallel section

#endif

      utils :: print_progress (i, num_batches, timer);

    } // end for

    std :: cout << std :: endl;
  }
}

void BasePlasticity :: _predict (const float * A, const float * B, float * C, const int & N, const int & M, const int & K)
{
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
