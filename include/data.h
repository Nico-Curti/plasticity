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

#ifndef __BaseData_loader__
#define __BaseData_loader__

#include <utils.hpp> // utility functions
#include <fstream> // file stream

#ifdef __view__

  #include <opencv2/core/mat.hpp>

#endif


namespace data_loader
{

/**
* @class BaseData
*
* @brief Base class for data loading
*
* @details The object allows the laod of the dataset
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
*/
class BaseData
{

public:

  int32_t num_train_sample; ///< number of training images/labels
  int32_t num_test_sample; ///< number of testing images/labels
  int32_t rows; ///< number of rows in each training/testing image
  int32_t cols; ///< number of columns in each training/testing image
  int32_t channels; ///< number of channels in each training/testing image

  std :: unique_ptr < uint8_t [] > training_images; ///< training image sequential buffer
  std :: unique_ptr < uint8_t [] > testing_images;  ///< testing image sequential buffer
  std :: unique_ptr < uint8_t [] > training_labels; ///< training label sequential buffer
  std :: unique_ptr < uint8_t [] > testing_labels;  ///< testing label sequential buffer

  // Constructor

  /**
  * @brief Default constructor
  */
  BaseData ();

  // Copy Operator and Copy Constructor

  /**
  * @brief Copy constructor.
  *
  * @details The copy constructor provides a deep copy of the object, i.e. all the
  * arrays are copied and not moved.
  *
  * @param x BaseData object
  *
  */
  BaseData (const BaseData & x);

  /**
  * @brief Copy operator.
  *
  * @details The operator performs a deep copy of the object and if there are buffers
  * already allocated, the operatore deletes them and then re-allocates an appropriated
  * portion of memory.
  *
  * @param x BaseData object
  *
  */
  BaseData & operator = (const BaseData & x);

  // Destructor

  /**
  * @brief Destructor.
  *
  * @details Completely delete the object and release the memory of the arrays.
  *
  */
  ~BaseData () = default;

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
  virtual void load_training_images (const std :: string & training_images) = 0;

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
  virtual void load_training_labels (const std :: string & training_labels) = 0;

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
  virtual void load_testing_images (const std :: string & testing_images) = 0;

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
  virtual void load_testing_labels (const std :: string & testing_labels) = 0;

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
  void load (const std :: string & training_images, const std :: string & training_labels,
    const std :: string & testing_images, const std :: string & testing_labels);

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

};


} // end namespace


#endif // __BaseData_loader__
