#include <base.h>

float BasePlasticity :: precision = 1e-30f;

BasePlasticity :: BasePlasticity () : optimizer (), w_init (), output (nullptr), weights (nullptr), history (), theta (nullptr), activation (nullptr), gradient (nullptr),
                                      batch (100), outputs (100), nweights (0), epochs_for_convergency (0), convergency_atol (0.f)
{
}

BasePlasticity :: BasePlasticity (const int & outputs, const int & batch_size, int activation,
                                  update_args optimizer,
                                  weights_initialization weights_init,
                                  int epochs_for_convergency, float convergency_atol
                                  ) : optimizer (optimizer), w_init (weights_init), output (nullptr), weights (nullptr), history (), theta (nullptr), activation (nullptr), gradient (nullptr),
                                      batch (batch_size), outputs (outputs), nweights (0), epochs_for_convergency (epochs_for_convergency), convergency_atol (convergency_atol)
{
  //// correct epochs_for_convergency
  ////this->epochs_for_convergency = std :: max(this->epochs_for_convergency, 1);

  this->activation = transfer :: activate( activation );
  this->gradient   = transfer :: gradient( activation );

  this->theta.reset(new float[this->outputs]);
}

BasePlasticity :: BasePlasticity (const BasePlasticity & b)
{
  this->nweights   = b.nweights;
  this->activation = b.activation;
  this->gradient   = b.gradient;

  this->batch    = b.batch;
  this->outputs  = b.outputs;
  this->nweights = b.nweights;
  this->epochs_for_convergency = b.epochs_for_convergency;

  this->convergency_atol = b.convergency_atol;

  this->optimizer = b.optimizer;
  this->w_init = b.w_init;

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
  this->epochs_for_convergency = b.epochs_for_convergency;

  this->convergency_atol = b.convergency_atol;

  this->optimizer = b.optimizer;
  this->w_init = b.w_init;

  this->weights.reset(new float[b.nweights]);
  std :: copy_n (b.weights.get(), b.nweights, this->weights.get());

  this->theta.reset(new float[b.outputs]);
  //std :: copy_n (b.theta.get(), b.outputs, this->theta.get()); // it is useless

  return *this;
}


void BasePlasticity :: fit (float * X, const int & n_samples, const int & n_features, const int & num_epochs, int seed)
{
  this->nweights = this->outputs * n_features;
  this->weights.reset(new float[this->nweights]);

  const int outputs = this->outputs * this->batch;
  this->output.reset(new float[outputs]);

  this->w_init.init(this->weights.get(), this->outputs, n_features);
  this->optimizer.init_arrays(this->nweights);

  this->_fit (X, num_epochs, n_features, n_samples, seed);
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

void BasePlasticity :: check_params ()
{
  if ( this->epochs_for_convergency <= 0 )
  {
    std :: cerr << "epochs_for_convergency must be an integer bigger or equal than 1" << std :: endl;
    std :: exit(ERROR_CONVERGENCY_POSITIVE);
  }
}

bool BasePlasticity :: check_convergency ()
{
  if ( static_cast < int >(this->history.size()) < this->epochs_for_convergency )
  {
    this->history.emplace_back(new float[this->outputs]);
    std :: copy_n(this->theta.get(), this->outputs, this->history[this->history.size() - 1].get());
    return false;
  }

  int check = 0;
  const float toll = this->convergency_atol;

  for (int i = 0; i < this->epochs_for_convergency; ++i)
  {
    check = std :: inner_product(this->theta.get(), this->theta.get() + this->outputs,
                                 this->history[i].get(), 0,
                                 std :: plus < float >(),
                                 [&](const float & theta, const float & history)
                                 {
                                   return static_cast < int >(std :: fabs(theta - history) < toll);
                                 });

    if ( check == this->outputs )
      goto stop;
  }

  this->history.pop_front();
  this->history.emplace_back(new float[this->outputs]);
  std :: copy_n(this->theta.get(), this->outputs, this->history[this->history.size() - 1].get());

  stop:
  return check == this->outputs;
}

void BasePlasticity :: normalize_weights ()
{
}


void BasePlasticity :: _fit (float * X, const int & num_epochs, const int & n_features, const int & n_samples, const int & seed)
{
  const int num_batches = n_samples / this->batch;

  std :: unique_ptr < float[] > weights_update (new float[this->nweights]);
  std :: unique_ptr < int[] > batch_indices(new int [num_batches]);
  std :: iota(batch_indices.get(), batch_indices.get() + num_batches, 0);

  std :: mt19937 engine(seed);

  for (int epoch = 0; epoch < num_epochs; ++epoch)
  {
    std :: shuffle(batch_indices.get(), batch_indices.get() + num_batches, engine);

#ifdef __verbose__

    std :: cout << RESET_COUT << "Epoch " << epoch + 1 << "/" << num_epochs << std :: endl;
    auto timer  = utils :: what_time_is_it_now();

#endif // __verbose__

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

        //const float epsil = 2e-2f * (1.f - epoch / num_epochs);
        //for (int i = 0; i < this->nweights; ++i)
        //  this->weights[i] += epsil * weights_update[i];
        this->optimizer.update(epoch + 1, this->weights.get(), weights_update.get(), this->nweights);

#ifdef _OPENMP

      } // end parallel section

#endif

#ifdef __verbose__

      utils :: print_progress (i + 1, num_batches, timer);

#endif // __verbose__

    } // end for batches

#ifdef __verbose__

    std :: cout << std :: endl;

#endif // __verbose__


    if ( this->check_convergency() )
    {

#ifdef __verbose__

      std :: cout << "Early stopping: the training has reached the convergency criteria" << std :: endl;

#endif // __verbose__

      break;
    }


  } // end for epoch
}

void BasePlasticity :: _predict (__unused const float * A, __unused const float * B, __unused float * C, __unused const int & N, __unused const int & M, __unused const int & K)
{
}
