#include <weights.h>

weights_initialization :: weights_initialization () : type (-1), mu (0.f), sigma (0.f), scale (0.f)
{
}

weights_initialization :: weights_initialization (const int & type, float mu, float sigma, float scale, int seed) : type (type), mu (mu), sigma (sigma), scale (scale)
{
  this->engine = std :: mt19937(seed);

  // check the initialization type (greater than zero and included into the get map of available values)
  if ( type < 0 || type >= static_cast < int >(weights_init :: get_weights.size()) )
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

weights_initialization :: weights_initialization (const weights_initialization & args) : engine (args.engine), type (args.type), mu (args.mu), sigma (args.sigma), scale (args.scale)
{
}

void weights_initialization :: init (float * weights, const int & inputs, const int & outputs)
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

void weights_initialization :: zeros (float * weights, const int & inputs, const int & outputs)
{
  const int size = inputs * outputs;

  std :: fill_n(weights, size, 0.f);
}

void weights_initialization :: ones (float * weights, const int & inputs, const int & outputs)
{
  const int size = inputs * outputs;

  std :: fill_n(weights, size, 1.f);
}

void weights_initialization :: uniform (float * weights, const int & inputs, const int & outputs)
{
  const int size = inputs * outputs;
  std :: uniform_real_distribution < float > random_uniform (-this->scale, this->scale);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_uniform(this->engine);
                     });
}

void weights_initialization :: normal (float * weights, const int & inputs, const int & outputs)
{
  const int size = inputs * outputs;
  std :: normal_distribution < float > random_normal (this->mu, this->sigma);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_normal(this->engine);
                     });
}

void weights_initialization :: lecun_uniform (float * weights, const int & inputs, const int & outputs)
{
  const int size = inputs * outputs;
  const float scale = math :: sqrt(3.f / inputs);
  std :: uniform_real_distribution < float > random_uniform (-scale, scale);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_uniform(this->engine);
                     });
}

void weights_initialization :: glorot_uniform (float * weights, const int & inputs, const int & outputs)
{
  const int size = inputs * outputs;
  const float scale = math :: sqrt(6.f / (inputs + outputs));
  std :: uniform_real_distribution < float > random_uniform (-scale, scale);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_uniform(this->engine);
                     });
}

void weights_initialization :: lecun_normal (float * weights, const int & inputs, const int & outputs)
{
  const int size = inputs * outputs;
  const float sigma = math :: sqrt(1.f / inputs);
  std :: normal_distribution < float > random_normal (this->mu, sigma);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_normal(this->engine);
                     });
}

void weights_initialization :: glorot_normal (float * weights, const int & inputs, const int & outputs)
{
  const int size = inputs * outputs;
  const float sigma = math :: sqrt(2.f / (inputs + outputs));
  std :: normal_distribution < float > random_normal (this->mu, sigma);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_normal(this->engine);
                     });
}

void weights_initialization :: he_uniform (float * weights, const int & inputs, const int & outputs)
{
  const int size = inputs * outputs;
  const float scale = math :: sqrt(6.f / inputs);
  std :: uniform_real_distribution < float > random_uniform (-scale, scale);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_uniform(this->engine);
                     });
}

void weights_initialization :: he_normal (float * weights, const int & inputs, const int & outputs)
{
  const int size = inputs * outputs;
  const float sigma = math :: sqrt(2.f / inputs);
  std :: normal_distribution < float > random_normal (this->mu, sigma);

  std :: generate_n (weights, size,
                     [&]()
                     {
                       return random_normal(this->engine);
                     });
}
