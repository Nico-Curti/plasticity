#include <data.h>


namespace data_loader
{

BaseData :: BaseData () : num_train_sample (0), num_test_sample (0), rows (0), cols (0), channels (0),
                          training_images (nullptr), testing_images (nullptr), training_labels (nullptr), testing_labels (nullptr)
{

}

BaseData :: BaseData (const BaseData & x) : num_train_sample (x.num_train_sample), num_test_sample (x.num_test_sample), rows (x.rows), cols (x.cols), channels (x.channels)
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


void BaseData :: load (const std :: string & training_images, const std :: string & training_labels, const std :: string & testing_images, const std :: string & testing_labels)
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

  return cv :: Mat(this->rows, this->cols, CV_MAKETYPE(CV_8U, (this->channels)), this->training_images.get() + start);
}

cv :: Mat BaseData :: get_test_image (const std :: size_t & idx)
{
  CV_Assert(static_cast < int32_t > (idx) < this->num_test_sample);

  // get the initial buffer position
  const int32_t start = idx * this->rows * this->cols * this->channels;

  return cv :: Mat(this->rows, this->cols, CV_MAKETYPE(CV_8U, (this->channels)), this->testing_images.get() + start);
}

#endif // __view__



} // end namespace
