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

#include <data.h>

namespace data_loader
{

BaseData :: BaseData () : num_train_sample (0), num_test_sample (0),
  rows (0), cols (0), channels (0),
  training_images (nullptr), testing_images (nullptr),
  training_labels (nullptr), testing_labels (nullptr)
{

}

BaseData :: BaseData (const BaseData & x) : num_train_sample (x.num_train_sample), num_test_sample (x.num_test_sample),
rows (x.rows), cols (x.cols), channels (x.channels)
{
  this->training_images.reset(new uint8_t[this->train_size()]);
  this->testing_images.reset(new uint8_t[this->test_size()]);

  this->training_labels.reset(new uint8_t[this->num_train_sample]);
  this->testing_labels.reset(new uint8_t[this->num_test_sample]);

  std :: copy_n(x.training_images.get(), this->train_size(), this->training_images.get());
  std :: copy_n(x.testing_images.get(), this->test_size(), this->testing_images.get());
  std :: copy_n(x.training_labels.get(), this->num_train_sample, this->training_labels.get());
  std :: copy_n(x.testing_labels.get(), this->num_test_sample, this->testing_labels.get());
}

BaseData & BaseData :: operator = (const BaseData & x)
{
  this->num_train_sample = x.num_train_sample;
  this->num_test_sample = x.num_test_sample;
  this->rows = x.rows;
  this->cols = x.cols;
  this->channels = x.channels;

  this->training_images.reset(new uint8_t[this->train_size()]);
  this->testing_images.reset(new uint8_t[this->test_size()]);

  this->training_labels.reset(new uint8_t[this->num_train_sample]);
  this->testing_labels.reset(new uint8_t[this->num_test_sample]);

  std :: copy_n(x.training_images.get(), this->train_size(), this->training_images.get());
  std :: copy_n(x.testing_images.get(), this->test_size(), this->testing_images.get());
  std :: copy_n(x.training_labels.get(), this->num_train_sample, this->training_labels.get());
  std :: copy_n(x.testing_labels.get(), this->num_test_sample, this->testing_labels.get());

  return *this;
}


void BaseData :: load (const std :: string & training_images, const std :: string & training_labels,
  const std :: string & testing_images, const std :: string & testing_labels)
{
  this->load_training_images(training_images);
  this->load_training_labels(training_labels);
  this->load_testing_images(testing_images);
  this->load_testing_labels(testing_labels);
}

int32_t BaseData :: train_size ()
{
  return this->num_train_sample * this->rows * this->cols * this->channels;
}

int32_t BaseData :: test_size ()
{
  return this->num_test_sample * this->rows * this->cols * this->channels;
}


#ifdef __view__

cv :: Mat BaseData :: get_train_image (const std :: size_t & idx)
{
  CV_Assert(static_cast < int32_t > (idx) < this->num_train_sample);

  // get the initial buffer position
  const int32_t start = idx * this->rows * this->cols * this->channels;

  return cv :: Mat(this->rows, this->cols,
                   CV_MAKETYPE(CV_8U, (this->channels)),
                   this->training_images.get() + start);
}

cv :: Mat BaseData :: get_test_image (const std :: size_t & idx)
{
  CV_Assert(static_cast < int32_t > (idx) < this->num_test_sample);

  // get the initial buffer position
  const int32_t start = idx * this->rows * this->cols * this->channels;

  return cv :: Mat(this->rows, this->cols,
                   CV_MAKETYPE(CV_8U, (this->channels)),
                   this->testing_images.get() + start);
}

#endif // __view__



} // end namespace
