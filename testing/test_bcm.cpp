#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <bcm.h>
#include <optimizer.h>
#include <Eigen/Dense>


#define PRECISION 1e-4f
#define SEED 42
#define isclose(x, y) ( std :: fabs((x) - (y)) < PRECISION )


TEST_CASE ( "Test fit" )
{
  std :: mt19937 engine(SEED);

  const int outputs = 10;
  const int batch_size = 10;
  const int activation = transfer :: _linear_;
  const float mu = 1.f;
  const float sigma = 2.5f;
  const float strenght = 0.f;
  const int seed = SEED;

  update_args optimizer(optimizer_t :: _sgd);

  BCM model(outputs, batch_size, activation, optimizer, mu, sigma, strenght, seed);

  const int num_epochs = 1;
  const int num_samples = batch_size;
  const int num_features = 5;

  std :: unique_ptr < float[] > data(new float[num_samples * num_features]);

  std :: normal_distribution < float > random_normal (0.f, 1.f);

  std :: generate_n (data.get(), num_samples * num_features,
                     [&]()
                     {
                       return random_normal(engine);
                     });

  REQUIRE_THROWS (model.predict(data.get(), num_samples, num_features), ERROR_FITTED);
  REQUIRE_THROWS (model.predict(data.get(), num_samples, num_samples), ERROR_DIMENSIONS);

  model.fit(data.get(), num_samples, num_features, num_epochs);

  SECTION ( "Test prediction" )
  {
    float * prediction = model.predict(data.get(), num_samples, num_features);

    Eigen :: Map < Eigen :: Matrix < float, outputs, num_features, Eigen :: RowMajor > > W(model.weights.get(), outputs, num_features);
    Eigen :: Map < Eigen :: Matrix < float, num_samples, num_features, Eigen :: RowMajor > > X(data.get(), num_samples, num_features);

    auto gt = W * X.transpose();

    for (int i = 0; i < outputs; ++i)
      for (int j = 0; j < batch_size; ++j)
      {
        const int idx = i * batch_size + j;
        REQUIRE(isclose(gt(i, j), prediction[idx]));
      }

  }


}
