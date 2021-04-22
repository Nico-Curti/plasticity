#include <mnist.h>


namespace data_loader
{

MNIST :: MNIST () : num_train_sample (0), num_test_sample (0), rows (0), cols (0),
                    training_images (nullptr), testing_images (nullptr), training_labels (nullptr), testing_labels (nullptr)
{

}

MNIST :: MNIST (const MNIST & x) : num_train_sample (x.num_train_sample), num_test_sample (x.num_test_sample), rows (x.rows), cols (x.cols)
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

MNIST & MNIST :: operator = (const MNIST & x)
{
  new (this) MNIST(x);
  return *this;
}


void MNIST :: load_training_images (const std :: string & training_images)
{
  // check paths
  if ( ! utils :: file_exists(training_images) )
    throw std :: runtime_error("Training image file not found. Given: " + training_images);

  // load the training image files
  this->load_file(training_images, this->training_images, MNIST_IMAGE_MAGIC_CODE, this->num_train_sample);
}

void MNIST :: load_training_labels (const std :: string & training_labels)
{
  // check paths
  if ( ! utils :: file_exists(training_labels) )
    throw std :: runtime_error("Training labels file not found. Given: " + training_labels);

  // load the training image/labels files
  this->load_file(training_labels, this->training_labels, MNIST_LABEL_MAGIC_CODE, this->num_train_sample);
}

void MNIST :: load_testing_images (const std :: string & testing_images)
{
  // check paths
  if ( ! utils :: file_exists(testing_images) )
    throw std :: runtime_error("Testing image file not found. Given: " + testing_images);

  // load the test image files
  this->load_file(testing_images, this->testing_images, MNIST_IMAGE_MAGIC_CODE, this->num_test_sample);
}

void MNIST :: load_testing_labels (const std :: string & testing_labels)
{
  // check paths
  if ( ! utils :: file_exists(testing_labels) )
    throw std :: runtime_error("Testing label file not found. Given: " + testing_labels);

  // load the test image files
  this->load_file(testing_labels, this->testing_labels, MNIST_LABEL_MAGIC_CODE, this->num_test_sample);
}

void MNIST :: load (const std :: string & training_images, const std :: string & training_labels, const std :: string & testing_images, const std :: string & testing_labels)
{
  this->load_training_images(training_images);
  this->load_training_labels(training_labels);
  this->load_testing_images(testing_images);
  this->load_testing_labels(testing_labels);
}

int32_t MNIST :: train_size ()
{
  return this->num_train_sample * this->rows * this->cols;
}

int32_t MNIST :: test_size ()
{
  return this->num_test_sample * this->rows * this->cols;
}

// Private members


void MNIST :: load_file (const std :: string & filename, std :: unique_ptr < uint8_t [] > & data, const uint32_t & magic_key, int32_t & nsample)
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


#ifdef __view__

cv :: Mat MNIST :: get_train_image (const std :: size_t & idx)
{
  CV_Assert(static_cast < int32_t > (idx) < this->num_train_sample);

  // get the initial buffer position
  const int32_t start = idx * this->rows * this->cols;

  return cv :: Mat(this->rows, this->cols, CV_8UC1, this->training_images.get() + start);
}

cv :: Mat MNIST :: get_test_image (const std :: size_t & idx)
{
  CV_Assert(static_cast < int32_t > (idx) < this->num_test_sample);

  // get the initial buffer position
  const int32_t start = idx * this->rows * this->cols;

  return cv :: Mat(this->rows, this->cols, CV_8UC1, this->testing_images.get() + start);
}

#endif // __view__


uint32_t MNIST :: read_header (char * buffer, const std :: size_t & position)
{
  uint32_t * header = reinterpret_cast < uint32_t * >(buffer);
  uint32_t value = *(header + position);

  return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}


} // end namespace
