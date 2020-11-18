#ifndef __data_dispatcher_h__
#define __data_dispatcher_h__

#include <utils.h>

#define ERROR_DISPATCH 104

/**
* @class data_dispatcher
*
* @brief Utility class for the subdivition of a matrix of data (stored in ravel format)
* into a series of batches.
*
* @details The object simply points to the original buffer of data without any copy.
* In this way the member function get_batch can return a portion of the original data
* according to the size given by the batch_dimensions member.
* The batches can be randomized and returned into a shuffle order using the
* variable shuffle in the constructor.
*
*/
class data_dispatcher
{
  float ** batch; ///< pointer matrix to the original data.

  std :: mt19937 engine; ///< Random number generator

  std :: unique_ptr < int[] > indices; ///< array of indices (it is necessary if shuffle is enabled)

public:

  int num_batches;  ///< number of batches
  int batch_dimension; ///< Dimension of the batch array (aka the lenght of the array returned by get_batch function).

  // Constructor

  /**
  * @brief Construct the data dispatcher.
  *
  * @details The constructor computes the data subdivisions and it
  * allocates the batches.
  *
  * @note The input data are not copied but the batch matrix points to the
  * positions of the original data.
  *
  * @param buffer Input array matrix in flatten format
  * @param batch_size Dimension of the batch
  * @param N_rows Original number of rows in the matrix
  * @param N_cols Original number of cols in the matrix
  * @param shuffle Enable/Disable the random shuffle of the input data
  * @param seed Random number generator seed.
  *
  */
  data_dispatcher (float * buffer, const int & batch_size,
                   const int & N_rows, const int & N_cols,
                   bool shuffle=true, int seed=42);


  // Destructor

  /**
  * @brief Destructor.
  *
  * @details Completely delete the object and release the memory of the arrays.
  *
  */
  ~data_dispatcher ();

  /**
  * @brief Get the required batch array.
  *
  * @details Return a reference to the original data matrix corresponding to the
  * required batch.
  *
  * @param idx Batch position.
  *
  * @return The subset of data in ravel format.
  *
  */
  float * get_batch (const int & idx);

  /**
  * @brief Apply a randomization of the batches.
  *
  * @details This function applies the same procedure of the constructor
  * in case of shuffle=true.
  *
  */
  void randomize ();

};


#endif // __data_dispatcher_h__
