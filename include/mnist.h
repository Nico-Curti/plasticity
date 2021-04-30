#ifndef __MNIST_loader__
#define __MNIST_loader__

#include <utils.hpp> // utility functions
#include <fstream> // file stream

#ifdef __view__

  #include <opencv2/core/mat.hpp>

#endif

#define MNIST_LABEL_MAGIC_CODE  0x801 ///< MNIST label magic code of the binary file
#define MNIST_IMAGE_MAGIC_CODE 0x803 ///< MNIST image magic code of the binary file


namespace data_loader
{

/**
* @class MNIST
*
* @brief Load the MNIST digit dataset
*
* @details The object allows the laod of the MNIST digit dataset
* from binary files. The training/testing images and labels are
* loaded into a sequential buffer of unsigned char (aka uint8_t)
* into 4 different arrays (2 arrays for the images and 2 for the labels).
* You can directly manage the sequential buffers (format required by the
* plasticity models) or, with the help of OpenCV (enabled with the __view__
* define), you can visualize each single image of the dataset using
* the corresponding index.
* The shape of the images as much as the number of samples are stored
* into the object.
*
* Reference: https://github.com/wichtounet/mnist
*
*/
class MNIST
{

public:

  int32_t num_train_sample; ///< number of training images/labels
  int32_t num_test_sample; ///< number of testing images/labels
  int32_t rows; ///< number of rows in each training/testing image
  int32_t cols; ///< number of columns in each training/testing image

  std :: unique_ptr < uint8_t [] > training_images; ///< training image sequential buffer
  std :: unique_ptr < uint8_t [] > testing_images;  ///< testing image sequential buffer
  std :: unique_ptr < uint8_t [] > training_labels; ///< training label sequential buffer
  std :: unique_ptr < uint8_t [] > testing_labels;  ///< testing label sequential buffer

  // Constructor

  /**
  * @brief Default constructor
  */
  MNIST ();

  // Copy Operator and Copy Constructor

  /**
  * @brief Copy constructor.
  *
  * @details The copy constructor provides a deep copy of the object, i.e. all the
  * arrays are copied and not moved.
  *
  * @param x MNIST object
  *
  */
  MNIST (const MNIST & x);

  /**
  * @brief Copy operator.
  *
  * @details The operator performs a deep copy of the object and if there are buffers
  * already allocated, the operatore deletes them and then re-allocates an appropriated
  * portion of memory.
  *
  * @param x MNIST object
  *
  */
  MNIST & operator = (const MNIST & x);

  // Destructor

  /**
  * @brief Destructor.
  *
  * @details Completely delete the object and release the memory of the arrays.
  *
  */
  ~MNIST () = default;

  /**
  * @brief Load the training image data from the binary files
  *
  * @details This function load and set the MNIST training images into the object buffers.
  * Internal variables of the object are set according to the bytes read from the
  * file. Some tests are internally performed to check the validity of the provided
  * files. A runtime error is raised if something goes wrong in the loading.
  *
  * @param training_images Filename/Path of the training image binary file.
  *
  */
  void load_training_images (const std :: string & training_images);

  /**
  * @brief Load the training labels data from the binary files
  *
  * @details This function load and set the MNIST training images into the object buffers.
  * Internal variables of the object are set according to the bytes read from the
  * file. Some tests are internally performed to check the validity of the provided
  * files. A runtime error is raised if something goes wrong in the loading.
  *
  * @param training_labels Filename/Path of the training labels binary file.
  *
  */
  void load_training_labels (const std :: string & training_labels);

  /**
  * @brief Load the testing image data from the binary files
  *
  * @details This function load and set the MNIST testing images into the object buffers.
  * Internal variables of the object are set according to the bytes read from the
  * file. Some tests are internally performed to check the validity of the provided
  * files. A runtime error is raised if something goes wrong in the loading.
  *
  * @param testing_images Filename/Path of the testing image binary file.
  *
  */
  void load_testing_images (const std :: string & testing_images);

  /**
  * @brief Load the testing labels data from the binary files
  *
  * @details This function load and set the MNIST testing images into the object buffers.
  * Internal variables of the object are set according to the bytes read from the
  * file. Some tests are internally performed to check the validity of the provided
  * files. A runtime error is raised if something goes wrong in the loading.
  *
  * @param testing_labels Filename/Path of the testing labels binary file.
  *
  */
  void load_testing_labels (const std :: string & testing_labels);

  /**
  * @brief Load the data from the binary files
  *
  * @details This function load and set the MNIST dataset into the object buffers.
  * Internal variables of the object are set according to the bytes read from the
  * file. Some tests are internally performed to check the validity of the provided
  * files. A runtime error is raised if something goes wrong in the loading.
  *
  * @param training_images Filename/Path of the training image binary file.
  * @param training_labels Filename/Path of the training label binary file.
  * @param testing_images Filename/Path of the testing image binary file.
  * @param testing_labels Filename/Path of the testing label binary file.
  *
  */
  void load (const std :: string & training_images, const std :: string & training_labels, const std :: string & testing_images, const std :: string & testing_labels);

#ifdef __view__

  /**
  * @brief Get the corresponding training image.
  *
  * @details This function allows the management of the training image
  * as series of OpenCV images. The provided index must be less than
  * the number of training samples stored.
  *
  * @param idx Index of the image to get.
  *
  * @return Correspoding OpenCV image.
  *
  */
  cv :: Mat get_train_image (const std :: size_t & idx);

  /**
  * @brief Get the corresponding testing image.
  *
  * @details This function allows the management of the testing image
  * as series of OpenCV images. The provided index must be less than
  * the number of testing samples stored.
  *
  * @param idx Index of the image to get.
  *
  * @return Correspoding OpenCV image.
  *
  */
  cv :: Mat get_test_image (const std :: size_t & idx);

#endif

  // Utility

  /**
  * @brief Get the training buffer size.
  *
  * @details The training size is equal to the number of training
  * samples multiplied by the number of rows and cols of the images.
  *
  * @return Training buffer size.
  */
  int32_t train_size ();

  /**
  * @brief Get the testing buffer size.
  *
  * @details The testing size is equal to the number of testing
  * samples multiplied by the number of rows and cols of the images.
  *
  * @return Training buffer size.
  */
  int32_t test_size ();

private:

  /**
  * @brief Core function of the file loading.
  *
  * @details This function allows the loading of both training and testing
  * binary files, setting the internal variables of the corresponding buffer.
  * The switch between training and test is performed by the magic_key provided.
  * For the label loading the magic key must be set according to the MNIST_LABEL_MAGIC_CODE,
  * while for the images the variable must be equal to MNIST_IMAGE_MAGIC_CODE.
  *
  * @param filename Filename/Path of the MNIST binary file.
  * @param buffer Corresponding buffer to fill.
  * @param magic_key Magic key for image/label loading.
  * @param nsample Number of training/test variable set into this function.
  *
  */
  void load_file (const std :: string & filename, std :: unique_ptr < uint8_t [] > & buffer, const uint32_t & magic_key, int32_t & nsample);

  /*!
  * @brief Extract the MNIST header from the given buffer
  *
  * @param buffer The current buffer
  * @param position The current reading positoin
  *
  * @return The value of the mnist header
  */
  uint32_t read_header (char * buffer, const std :: size_t & position);

};


} // end namespace


#endif // __MNIST_loader__
