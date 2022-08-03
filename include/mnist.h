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

#ifndef __MNIST_loader__
#define __MNIST_loader__

#include <data.h> // BaseData class

#define MNIST_LABEL_MAGIC_CODE 0x801 ///< MNIST label magic code of the binary file
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
class MNIST : public BaseData
{

public:

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
  void load_file (const std :: string & filename, std :: unique_ptr < uint8_t [] > & buffer,
    const uint32_t & magic_key, int32_t & nsample);

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
