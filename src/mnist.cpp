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

#include <mnist.h>

namespace data_loader
{

MNIST :: MNIST () : BaseData ()
{
}

MNIST :: MNIST (const MNIST & x) : BaseData (x)
{
}

MNIST & MNIST :: operator = (const MNIST & x)
{
  BaseData :: operator = (x);
  return *this;
}

void MNIST :: load_training_images (const std :: string & training_images)
{
  // check paths
  if ( ! utils :: file_exists(training_images) )
    throw std :: runtime_error("Training image file not found. Given: " +
                               training_images);

  // load the training image files
  this->load_file(training_images, this->training_images,
    MNIST_IMAGE_MAGIC_CODE, this->num_train_sample);
}

void MNIST :: load_training_labels (const std :: string & training_labels)
{
  // check paths
  if ( ! utils :: file_exists(training_labels) )
    throw std :: runtime_error("Training labels file not found. Given: " +
                               training_labels);

  // load the training image/labels files
  this->load_file(training_labels, this->training_labels,
    MNIST_LABEL_MAGIC_CODE, this->num_train_sample);
}

void MNIST :: load_testing_images (const std :: string & testing_images)
{
  // check paths
  if ( ! utils :: file_exists(testing_images) )
    throw std :: runtime_error("Testing image file not found. Given: " +
                               testing_images);

  // load the test image files
  this->load_file(testing_images, this->testing_images,
    MNIST_IMAGE_MAGIC_CODE, this->num_test_sample);
}

void MNIST :: load_testing_labels (const std :: string & testing_labels)
{
  // check paths
  if ( ! utils :: file_exists(testing_labels) )
    throw std :: runtime_error("Testing label file not found. Given: " +
                               testing_labels);

  // load the test image files
  this->load_file(testing_labels, this->testing_labels,
    MNIST_LABEL_MAGIC_CODE, this->num_test_sample);
}

// Private members


void MNIST :: load_file (const std :: string & filename, std :: unique_ptr < uint8_t [] > & data,
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

  // start the processing
  uint32_t key = this->read_header(buffer.get(), 0);

  if (key != magic_key)
    throw std :: runtime_error ("Invalid magic number, probably not a MNIST file");

  // read the number of files
  nsample = this->read_header(buffer.get(), 1);

  // check the file validity
  switch (key)
  {
    case MNIST_IMAGE_MAGIC_CODE:
    {
      // read the image dimensions
      this->rows = this->read_header(buffer.get(), 2);
      this->cols = this->read_header(buffer.get(), 3);
      this->channels = 1; // single channel images (aka gray-scale)

      // consistency check
      if (size < nsample * this->rows * this->cols + 16)
        throw std :: runtime_error ("The file is not large enough to hold all the data, probably corrupted");

      // read the buffer of data
      data.reset(new uint8_t[size - 16]); // 16 is the size of the already read bytes
      std :: move(buffer.get() + 16, buffer.get() + size - 16, data.get());

    } break;

    case MNIST_LABEL_MAGIC_CODE:
    {
      // consistency check
      if (size < nsample + 8)
        throw std :: runtime_error ("The file is not large enough to hold all the data, probably corrupted");

      // read the buffer of data
      data.reset(new uint8_t[size - 8]); // 8 is the size of the already read bytes
      std :: move(buffer.get() + 8, buffer.get() + size - 8, data.get());

    } break;
  }

}

uint32_t MNIST :: read_header (char * buffer, const std :: size_t & position)
{
  uint32_t * header = reinterpret_cast < uint32_t * >(buffer);
  uint32_t value = *(header + position);

  return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}


} // end namespace
