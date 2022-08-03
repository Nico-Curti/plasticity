/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  The OpenHiP package is licensed under the MIT "Expat" License:
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

namespace data_loader
{

CIFAR10 :: CIFAR10 () : BaseData ()
{
}

CIFAR10 :: CIFAR10 (const CIFAR10 & x) : BaseData (x)
{
}

CIFAR10 & CIFAR10 :: operator = (const CIFAR10 & x)
{
  BaseData :: operator = (x);
  return *this;
}

void CIFAR10 :: load_training_images (const std :: string & training_images)
{
  // check paths
  if ( ! utils :: file_exists(training_images) )
    throw std :: runtime_error("Training image file not found. Given: " +
                               training_images);

  // load the training image files
  this->load_file(training_images, this->training_images,
    CIFAR10_IMAGE_MAGIC_CODE, this->num_train_sample);
}

void CIFAR10 :: load_training_labels (const std :: string & training_labels)
{
  // check paths
  if ( ! utils :: file_exists(training_labels) )
    throw std :: runtime_error("Training labels file not found. Given: " +
                               training_labels);

  // load the training image/labels files
  this->load_file(training_labels, this->training_labels,
    CIFAR10_LABEL_MAGIC_CODE, this->num_train_sample);
}

void CIFAR10 :: load_testing_images (const std :: string & testing_images)
{
  // check paths
  if ( ! utils :: file_exists(testing_images) )
    throw std :: runtime_error("Testing image file not found. Given: " +
                               testing_images);

  // load the test image files
  this->load_file(testing_images, this->testing_images,
    CIFAR10_IMAGE_MAGIC_CODE, this->num_test_sample);
}

void CIFAR10 :: load_testing_labels (const std :: string & testing_labels)
{
  // check paths
  if ( ! utils :: file_exists(testing_labels) )
    throw std :: runtime_error("Testing label file not found. Given: " +
                               testing_labels);

  // load the test image files
  this->load_file(testing_labels, this->testing_labels,
    CIFAR10_LABEL_MAGIC_CODE, this->num_test_sample);
}


// Private members

void CIFAR10 :: load_file (const std :: string & filename, std :: unique_ptr < uint8_t [] > & data,
  const uint32_t & magic_key, int32_t & nsample)
{
  // open the file
  std :: ifstream is(filename, std :: ios :: in | std :: ios :: binary | std :: ios :: ate);

  // determine the file size
  std :: streamsize size = is.tellg();
  // reset the stream position
  is.seekg(0, std :: ios :: beg);
  // allocate the global buffer
  std :: unique_ptr < char [] > buffer(new char[size]);
  // read the file content
  is.read(buffer.get(), size);

  // close the file
  is.close();

  const int32_t image_size = 32 * 32 * 3;

  // read the number of files
  nsample = size / (image_size + 1); // 32x32x3 images + 1 byte label

  // start the processing

  switch (magic_key)
  {
    case CIFAR10_IMAGE_MAGIC_CODE:
    {
      // read the image dimensions
      this->rows = 32;
      this->cols = 32;
      this->channels = 3; // 3-channels images (aka RGB)

      // read the buffer of data
      data.reset(new uint8_t[size - nsample]); // nsample is the number of labels

      for (int32_t i = 0; i < nsample; ++i)
        std :: move(buffer.get() + i * image_size + i + 1,
                    buffer.get() + i * image_size + i + 1 + image_size,
                    data.get() + i * image_size);

    } break;

    case CIFAR10_LABEL_MAGIC_CODE:
    {
      // read the buffer of data
      data.reset(new uint8_t[nsample]);

      for (int32_t i = 0; i < nsample; ++i)
        data[i] = buffer[i * (image_size + 1)];

    } break;
  }

}


} // end namespace
