/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  The plasticity package is licensed under the MIT "Expat" License:
//
//  Copyright (c) 2021: Nico Curti.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  the software is provided "as is", without warranty of any kind, express or
//  implied, including but not limited to the warranties of merchantability,
//  fitness for a particular purpose and noninfringement. in no event shall the
//  authors or copyright holders be liable for any claim, damages or other
//  liability, whether in an action of contract, tort or otherwise, arising from,
//  out of or in connection with the software or the use or other dealings in the
//  software.
//
//M*/

#include <mnist.h>
#include <hopfield.h>
#include <parser.h>

#ifdef __view__

  #include <opencv2/core.hpp>
  #include <opencv2/core/eigen.hpp>
  #include <opencv2/imgproc.hpp>
  #include <opencv2/highgui.hpp>

  #include <cmap.h>

#endif // __view__


void usage (char ** argv)
{
  std :: cerr << "Error parsing! Configuration file not provided" << std :: endl;
  std :: cerr << "Hopfield simulator with MNIST dataset" << std :: endl;
  std :: cerr << "Usage: " << argv[0] << " [config_filename]" << std :: endl;
  std :: cerr << "\t- config_filename : Configuration file with simulation parameters" << std :: endl;
  std :: exit (1);
}


int main (int argc, char ** argv)
{

  if (argc <= 1)
    usage(argv);

  const std :: string data_cfg = std :: string(argv[1]);

  /******************************************************
                Read Configuration Informations
  *******************************************************/

  parser :: config cfg (data_cfg);

  const std :: string training_file = cfg.get < std :: string > ("MNIST_training_image", "");

  const int outputs                = cfg.get < int > ("outputs", 100);
  const int batch_size             = cfg.get < int > ("batch_size", 1000);
  const int epochs_for_convergency = cfg.get < int > ("epochs_for_convergency", 100000);
  const float convergency_atol     = cfg.get < float > ("convergency_atol", 1e10f);
  //const float interaction_strength = cfg.get < float > ("interaction_strength", 0.0f);
  const int seed                   = cfg.get < int > ("seed", 42);
  const int epochs                 = cfg.get < int > ("epochs", 1000);

  const float delta                = cfg.get < float > ("delta", .4f);
  const float p                    = cfg.get < float > ("p", 2.f);
  const int k                      = cfg.get < int > ("k", 2);
  //const int activation_func        = transfer :: get_activation.at(cfg.get < std :: string > ("activation", "logistic"));
  const float weights_decay        = cfg.get < float > ("weights_decay", 0.f);

  const int optimizer_type         = optimizer :: get_optimizer.at(cfg.get < std :: string > ("optimizer", "sgd"));
  const float learning_rate        = cfg.get < float > ("learning_rate", 2e-2f);
  const float momentum             = cfg.get < float > ("momentum", .9f);
  const float decay                = cfg.get < float > ("decay", 1e-4f);
  const float B1                   = cfg.get < float > ("B1", .9f);
  const float B2                   = cfg.get < float > ("B2", .999f);
  const float rho                  = cfg.get < float > ("rho", 0.f);

  const int weights_type           = weights_init :: get_weights.at(cfg.get < std :: string > ("weights", "normal"));
  const float mu                   = cfg.get < float > ("mu", 0.f);
  const float std                  = cfg.get < float > ("std", 1.f);
  const float scale                = cfg.get < float > ("scale", 1.f);
  const int weights_seed           = cfg.get < int > ("weights_seed", 42);

  const int normalize              = cfg.get < int > ("normalize", 1);
  const int binarize               = cfg.get < int > ("binarize", 0);

  /******************************************************
                Load the MNIST training set
  *******************************************************/

  data_loader :: MNIST dataset;
  dataset.load_training_images(training_file);

  std :: cout << "Dataset parameters:" << std :: endl;
  std :: cout << "- Number of training images : " << dataset.num_train_sample << std :: endl;
  std :: cout << "- Image size : [" << dataset.rows << ", " << dataset.cols << ", " << dataset.channels << "]" << std :: endl;

  /******************************************************
                Preprocess the data
  *******************************************************/

  std :: unique_ptr < float [] > training(new float[dataset.train_size()]);

  for (int i = 0; i < dataset.train_size(); ++i)
  {
    training[i] = static_cast < float >(dataset.training_images[i]);

    if (normalize)
      training[i] /= 255.f;
    else

    if (binarize)
      training[i] = static_cast < float >(training[i] != 0);
  }

  /******************************************************
                Build the model
  *******************************************************/

  std :: cout << "Model parameters:" << std :: endl;
  std :: cout << "- Neurons: " << outputs << std:: endl;
  std :: cout << "- Epochs: " << epochs << std:: endl;
  std :: cout << "- Batch size: " << batch_size << std:: endl;
  //std :: cout << "- Lateral interaction strength: " << interaction_strength << std:: endl;
  //std :: cout << "- Activation Function: " << cfg.get < std :: string > ("activation", "logistic (default)") << std:: endl;
  std :: cout << "- Delta: " << delta << std:: endl;
  std :: cout << "- P: " << p << std:: endl;
  std :: cout << "- K: " << k << std:: endl;
  std :: cout << "- Weights decay: " << weights_decay << std:: endl;
  std :: cout << "- Weights Model: " << cfg.get < std :: string > ("weights", "normal (default)") << std:: endl;
  std :: cout << "  - Mean: " << mu << std :: endl;
  std :: cout << "  - Std: " << std << std :: endl;
  std :: cout << "  - Scale: " << scale << std :: endl;
  std :: cout << "  - Seed: " << weights_seed << std :: endl;
  std :: cout << "- Optimizer: " << cfg.get < std :: string > ("optimizer", "sgd (default)") << std:: endl;
  std :: cout << "  - Learning rate: " << learning_rate << std :: endl;
  std :: cout << "  - Momentum: " << momentum << std :: endl;
  std :: cout << "  - Decay: " << decay << std :: endl;
  std :: cout << "  - B1: " << B1 << std :: endl;
  std :: cout << "  - B2: " << B2 << std :: endl;
  std :: cout << "  - rho: " << rho << std :: endl;


  Hopfield hopfield (outputs, batch_size,
                     update_args(optimizer_type, learning_rate, momentum, decay, B1, B2, rho),
                     weights_initialization(weights_type, mu, std, scale, weights_seed),
                     epochs_for_convergency, convergency_atol,
                     decay, delta, p, k);

#ifdef __view__

  /******************************************************
                Set view callback
  *******************************************************/

  cv :: namedWindow("Learning weights", cv :: WINDOW_FULLSCREEN);
  cv :: moveWindow("Learning weights", 20, 20);

  int32_t iter = 0;

  auto callback = [&](BasePlasticity * hopfield) -> void
                  {
                    // set the maximum number of images/neurons available in a square matrix
                    const int32_t num_images = std :: sqrt(hopfield->weights.rows());
                    // get the number image size
                    const int32_t size = dataset.rows;

                    // this will be the output image
                    cv :: Mat display;

                    // for each neuron we will concatenate the corresponding weights
                    // to build a square image
                    for (int32_t i = 0; i < num_images; ++i)
                    {
                      cv :: Mat row;

                      for (int32_t j = 0; j < num_images; ++j)
                      {
                        const int32_t index = i * num_images + j;

                        // get the correspoding block matrix
                        Eigen :: MatrixXf block = hopfield->weights.row(index);
                        // convert it to an OpenCV mat
                        cv :: Mat img;
                        cv :: eigen2cv(block, img);

                        // normalize each neuron independently (?)
                        cv :: normalize(img, img, 0., 255., cv :: NORM_MINMAX);

                        // reshape the image
                        img.convertTo(img, CV_8UC1);
                        img = img.reshape(1, size);

                        if (j == 0)
                          row = img.clone();
                        else
                          cv :: hconcat(row, img, row);
                      }

                      if (i == 0)
                        display = row.clone();
                      else
                        cv :: vconcat(display, row, display);
                    }

                    // normalize all the neurons (?)
                    //cv :: normalize(display, display, 0., 255., cv :: NORM_MINMAX);
                    ++ iter;

                    // apply the CUSTOM color map
                    cv :: applyColorMap(display, display, cv :: COLORMAP_BWR);
                    // resize the image to a reasonable size
                    cv :: resize(display, display, cv :: Size(512, 512), 0., 0., cv :: INTER_CUBIC);

                    // set the window name according the the number of updates performed
                    cv :: setWindowTitle("Learning weights", "Learning weights (it: " + std :: to_string(iter) + ")");

                    // visualize the image
                    cv :: resizeWindow("Learning weights", display.rows, display.cols);
                    cv :: imshow("Learning weights", display);
                    cv :: waitKey(1);
                  };

  /******************************************************
                Run the simulation
  *******************************************************/

  hopfield.fit(training.get(), dataset.num_train_sample, dataset.rows * dataset.cols * dataset.channels, epochs, seed, callback);

  // stop the image until the ESC key
  std :: cerr << "Press ESC to exit" << std :: endl;
  while ((cv :: waitKey(0) & 0xEFFFFF) != 27); //27 is the keycode for ESC
  cv :: destroyWindow("Learning weights");

#else

  /******************************************************
                Run the simulation
  *******************************************************/

  hopfield.fit(training.get(), dataset.num_train_sample, dataset.rows * dataset.cols * dataset.channels, epochs, seed);

#endif

  return 0;
}
