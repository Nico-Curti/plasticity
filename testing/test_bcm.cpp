#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <bcm.h>


#define PRECISION 1e-4f
#define SEED 42
#define isclose(x, y) ( std :: fabs((x) - (y)) < PRECISION )

std :: mt19937 engine(SEED);


TEST_CASE ( "Constructor" )
{
  const int32_t outputs = 10;
  const int32_t batch_size = 10;
  const int32_t activation = transfer_t :: linear;
  const float strenght = 0.f;

  update_args optimizer(optimizer_t :: sgd);
  weights_initialization weights_init(weights_init_t :: normal);

  BCM model(outputs, batch_size, activation, optimizer, weights_init, 1., 1e-2f, strenght);

  REQUIRE (model.weights.rows() == 0);
  REQUIRE (model.weights.cols() == 0);
}


TEST_CASE ( "Save/Load weights" )
{
  const int32_t outputs = 10;
  const int32_t batch_size = 10;
  const int32_t activation = transfer_t :: linear;
  const float strenght = 0.f;

  update_args optimizer(optimizer_t :: sgd);
  weights_initialization weights_init(weights_init_t :: normal);

  BCM model(outputs, batch_size, activation, optimizer, weights_init, 1., 1e-2f, strenght);

  REQUIRE_THROWS_AS (model.save_weights("dummy"), std :: runtime_error);
  REQUIRE_THROWS_WITH (model.save_weights("dummy"),
    "Fitted error. The model is not fitted yet.\n"
    "Please call the fit function before using the predict member.");

  const int32_t num_epochs = 1;
  const int32_t num_samples = batch_size;
  const int32_t num_features = 5;

  std :: unique_ptr < float[] > data(new float[num_samples * num_features]);

  std :: normal_distribution < float > random_normal (0.f, 1.f);

  std :: generate_n (data.get(), num_samples * num_features,
                     [&]()
                     {
                       return random_normal(engine);
                     });

  model.fit(data.get(), num_samples, num_features, num_epochs);

  REQUIRE (model.weights.rows() == outputs);
  REQUIRE (model.weights.cols() == num_features);

  REQUIRE_THROWS_AS (model.load_weights("dummy"), std :: runtime_error);
  REQUIRE_THROWS_WITH (model.load_weights("dummy"), "File not found. Given : dummy");

  Eigen :: MatrixXf Winit = model.weights;

  model.save_weights("dummy.bin");
  model.load_weights("dummy.bin");

  Eigen :: MatrixXf Wafter = model.weights;

  REQUIRE (Winit.isApprox(Wafter, PRECISION));
}


TEST_CASE ( "Fale prediction" )
{
  const int32_t outputs = 10;
  const int32_t batch_size = 10;
  const int32_t activation = transfer_t :: linear;
  const float strenght = 0.f;

  update_args optimizer(optimizer_t :: sgd);
  weights_initialization weights_init(weights_init_t :: normal);

  BCM model(outputs, batch_size, activation, optimizer, weights_init, 1., 1e-2f, strenght);

  const int32_t num_epochs = 1;
  const int32_t num_samples = batch_size;
  const int32_t num_features = 5;

  std :: unique_ptr < float[] > data(new float[num_samples * num_features]);

  std :: normal_distribution < float > random_normal (0.f, 1.f);

  std :: generate_n (data.get(), num_samples * num_features,
                     [&]()
                     {
                       return random_normal(engine);
                     });

  REQUIRE_THROWS_WITH (model.predict(data.get(), num_samples, num_features),
    "Fitted error. The model is not fitted yet.\n"
    "Please call the fit function before using the predict member.");

  model.fit(data.get(), num_samples, num_features, num_epochs);
  REQUIRE (model.weights.rows() == outputs);
  REQUIRE (model.weights.cols() == num_features);

  REQUIRE_THROWS_WITH (model.predict(data.get(), num_samples, num_samples),
    "Invalid dimensions found. The input (n_samples, n_features)"
    "shape is inconsistent with the number of weights (" +
    std :: to_string(outputs * num_features) + ")");
}


TEST_CASE ( "Fit buffer" )
{
  const int32_t outputs = 10;
  const int32_t batch_size = 10;
  const int32_t activation = transfer_t :: linear;
  const float strenght = 0.f;

  update_args optimizer(optimizer_t :: sgd);
  weights_initialization weights_init(weights_init_t :: normal);

  BCM model(outputs, batch_size, activation, optimizer, weights_init, 1., 1e-2f, strenght);

  const int32_t num_epochs = 1;
  const int32_t num_samples = batch_size;
  const int32_t num_features = 5;

  std :: unique_ptr < float[] > data(new float[num_samples * num_features]);

  std :: normal_distribution < float > random_normal (0.f, 1.f);

  std :: generate_n (data.get(), num_samples * num_features,
                     [&]()
                     {
                       return random_normal(engine);
                     });

  model.fit(data.get(), num_samples, num_features, num_epochs);

  REQUIRE (model.weights.rows() == outputs);
  REQUIRE (model.weights.cols() == num_features);

}


TEST_CASE ( "Predict" )
{
  const int32_t outputs = 10;
  const int32_t batch_size = 10;
  const int32_t activation = transfer_t :: linear;
  const float strenght = 0.f;

  update_args optimizer(optimizer_t :: sgd);
  weights_initialization weights_init(weights_init_t :: normal);

  BCM model(outputs, batch_size, activation, optimizer, weights_init, 1., 1e-2f, strenght);

  const int32_t num_epochs = 1;
  const int32_t num_samples = batch_size;
  const int32_t num_features = 5;

  std :: unique_ptr < float[] > data(new float[num_samples * num_features]);

  std :: normal_distribution < float > random_normal (0.f, 1.f);

  std :: generate_n (data.get(), num_samples * num_features,
                     [&]()
                     {
                       return random_normal(engine);
                     });


  model.fit(data.get(), num_samples, num_features, num_epochs);

  float * output_ptr = model.predict(data.get(), num_samples, num_features);
  Eigen :: Map < Eigen :: Matrix < float, outputs, num_samples, Eigen :: RowMajor > > output(output_ptr, outputs, num_samples);

  REQUIRE (output.rows() == outputs);
  REQUIRE (output.cols() == num_samples);
}

