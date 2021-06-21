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

#include <cifar10.h>
#include <parser.h>

#ifdef __view__

  #include <opencv2/imgproc.hpp>
  #include <opencv2/highgui.hpp>

#endif // __view__


void usage (char ** argv)
{
  std :: cerr << "Error parsing! Configuration file not provided" << std :: endl;
  std :: cerr << "BCM simulator with CIFAR-10 dataset" << std :: endl;
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

  const std :: string training_file       = cfg.get < std :: string > ("CIFAR10_training_image", "");
  const std :: string training_label_file = cfg.get < std :: string > ("CIFAR10_training_label", "");
  const std :: string testing_file        = cfg.get < std :: string > ("CIFAR10_testing_image", "");
  const std :: string testing_label_file  = cfg.get < std :: string > ("CIFAR10_testing_label", "");

  /******************************************************
              Load the CIFAR-10 training set
  *******************************************************/

  data_loader :: CIFAR10 dataset;
  dataset.load_training_images(training_file);
  dataset.load_testing_images(testing_file);
  dataset.load_training_labels(training_label_file);
  dataset.load_testing_labels(testing_label_file);
  // the lines above can be substituted by the following one
  //dataset.load(training_file, training_label_file, testing_file, testing_label_file);

  std :: cout << "Dataset parameters:" << std :: endl;
  std :: cout << "- Number of training images : " << dataset.num_train_sample << std :: endl;
  std :: cout << "- Number of testing images : " << dataset.num_test_sample << std :: endl;
  std :: cout << "- Image size : [" << dataset.rows << ", " << dataset.cols << "]" << std :: endl;

  /******************************************************
                    View some images
  *******************************************************/


#ifdef __view__

  const std :: string window_name = "CIFAR-10 dataset";
  cv :: namedWindow(window_name, cv :: WINDOW_FULLSCREEN);
  cv :: moveWindow(window_name, 20, 20);


  // set the number of images to display in a square matrix
  const int32_t num_images = 10;

  // this will be the output image
  cv :: Mat display;

  for (int32_t i = 0; i < num_images; ++i)
  {
    cv :: Mat row;

    for (int32_t j = 0; j < num_images; ++j)
    {
      const int32_t index = i * num_images + j;

      // extract the corresponding image
      cv :: Mat img = dataset.get_train_image(index);

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

  // resize the image to a reasonable size
  cv :: resize(display, display, cv :: Size(512, 512), 0., 0., cv :: INTER_CUBIC);
  cv :: cvtColor(display, display, cv :: COLOR_RGB2BGR);

  // visualize the image
  cv :: resizeWindow(window_name, display.rows, display.cols);
  cv :: imshow(window_name, display);
  cv :: waitKey(0);

#endif // __view__


  return 0;
}