#include <hopfield.h>

int main (__unused int argc, __unused char ** argv)
{

  const int n_samples = 1000;
  const int n_features = 100;

  std :: unique_ptr < float[] > data(new float[n_samples * n_features]);
  std :: iota(data.get(), data.get() + n_samples * n_features, 0.f);

  Hopfield hopfield (10, 4);
  hopfield.fit(data.get(), n_samples, n_features, 10);

  return 0;
}

