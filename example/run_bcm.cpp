#include <bcm.h>

int main (int argc, char ** argv)
{

  const int n_samples = 10;
  const int n_features = 100;

  std :: unique_ptr < float[] > data(new float[n_samples * n_features]);

  std :: normal_distribution < float > random_normal (0.f, 1.f);
  std :: generate_n (data.get(), n_samples * n_features,
                     [&]()
                     {
                       return random_normal(BasePlasticity :: engine);
                     });


  const int outputs = 10;
  const int batch_size = 4;
  const float mu = 0.f;
  const float sigma = 1.f;
  const int epochs_for_convergency = 1;
  const float convergency_atol = 1e10f;
  const float interaction_strength = 0.f;
  const int seed = 42;

  const int epochs = 100;

  BCM bcm (outputs, batch_size, transfer :: _logistic_, update_args(optimizer_t :: _sgd), mu, sigma,
           epochs_for_convergency, convergency_atol, interaction_strength, seed);
  bcm.fit(data.get(), n_samples, n_features, epochs);

  return 0;
}

