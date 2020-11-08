#ifndef __utils_h__
#define __utils_h__

#include <cmath>     // M_PI
#include <climits>   // std :: numeric_limits
#include <random>    // std :: mt19937
#include <iostream>  // std :: cout
#include <iomanip>   // std :: setprecision
#include <algorithm> // std :: sort       (if necessary)
#include <memory>    // std :: unique_ptr (if necessary)
#include <string>    // MSVC compatibility

#include <fmath.h>   // fast math function

#if defined _OPENMP

  #include <omp.h>      // omp_get_num_threads, timing function

  #include <utility>    // std :: pair for argsort
  #include <functional> // std :: greater

#elif defined _MPI

  #include <boost/mpi/timer.hpp> // boost :: mpi :: timer()

#else

  #include <chrono>     // timing function

#endif

#ifdef DEBUG

  #include <cassert>

#endif

// Usefull macro

#ifdef _MSC_VER

  #ifndef __unused
    #define __unused
  #endif

  #ifdef LINKING_DLL // necessary for export of some symbols
    #define DLLSPEC __declspec(dllimport)
  #else
    #define DLLSPEC __declspec(dllexport)
  #endif

#else // Not Visual Studio Compiler

  #ifndef __unused
    #define __unused __attribute__((__unused__))
  #endif

  #define DLLSPEC

#endif


// Usefull variables

#define PBWIDTH 20
static const std :: string FILL_VALUE = "█";
static const std :: string NULL_VALUE = " ";

#ifdef _WIN32
  #define RESET_COUT '\r'
#else
  #define RESET_COUT "\r\x1B[K"
#endif


// Usefull common variables

namespace utils
{

  // Timing functions

#if defined _OPENMP

  /**
  * @brief Return the current time (OpenMP format).
  *
  * @details There are many specialization of this function according to different parallel environments.
  *
  * @return The current time as double value.
  *
  */
  double what_time_is_it_now ();

#elif defined _MPI

  /**
  * @brief Return the current time (MPI format).
  *
  * @details There are many specialization of this function according to different parallel environments.
  *
  * @return The current time as MPI timer.
  *
  */
  boost :: mpi :: timer what_time_is_it_now ();

#else

  /**
  * @brief Return the current time (STD format).
  *
  * @details There are many specialization of this function according to different parallel environments.
  *
  * @return The current time as chrono timer value.
  *
  */
  std :: chrono :: time_point < std :: chrono :: high_resolution_clock > what_time_is_it_now ();

#endif

  /**
  * @brief Get the elapsed time from the start.
  *
  * @details This function is related to the what_time_is_it_now function and thus
  * its templates must be set according to the output of the what_time_is_it_now function.
  *
  * @tparam Time Timer type returned by the what_time_is_it_now function
  * @param start_time Start time.
  *
  * @return The elapsed time as double.
  *
  */
  template < typename Time >
  double elapsed_time (const Time & start_time);

  template < typename Char, typename Traits, typename Allocator >
  std :: basic_string < Char, Traits, Allocator > operator * (const std :: basic_string < Char, Traits, Allocator > & s, std :: size_t n);

  template < typename Char, typename Traits, typename Allocator >
  std :: basic_string < Char, Traits, Allocator > operator * (std :: size_t n, const std :: basic_string < Char, Traits, Allocator > & s);

  template < typename Time >
  void print_progress (const int & i, const int & num_batches, Time & timer);

  // OS functions

  /**
  * @brief Check if the given file exists.
  *
  * @details This function tries to open the given file (portable solution)
  * and return true/false according to the outcome of this action.
  *
  * @param filename Filename or path to check.
  *
  * @return True if the file exists. False otherwise.
  *
  */
  bool file_exists (const std :: string & filename);

#ifdef _OPENMP

  // parallel merge argsort

  template < typename lambda >
  void mergeargsort_serial ( std :: pair < float, int > * array, const float * a, const int & start, const int & end, lambda ord);

  template < typename lambda >
  void mergeargsort_parallel_omp ( std :: pair < float, int > * array, const float * a, const int & start, const int & end, const int & threads, lambda ord);

  template < typename lambda >
  void argsort (const float * a, int * indexes, const int & start, const int & end, lambda ord);

#endif

} // end namespace utils


#if __cplusplus < 201700 && !defined _MSC_VER // no std=c++17 support

namespace std
{


/**
* @brief Clamp the given value between the two extrema
*
* @details C++11 compatible implementation of clap function with template
* values.
*
* @tparam type Value data type
* @param v Input value.
* @param lo Low value of the domain.
* @param hi High value of the domain.
*
* @return Value clamped between the two extrema.
*
*/
template < class type >
const type & clamp ( const type & v, const type & lo, const type & hi );

#if ( ( __cplusplus < 201100 && !(_MSC_VER) ) || ( __GNUC__ == 4 && __GNUC_MINOR__ < 9) && !(__clang__) )
/**
* @brief Retro-compatibility solution for smart-pointers.
*
* @details C++11 compatible implementation of make_unique function for unique_ptr
* with array as template.
*
* @tparam type Array data type.
* @param size Length of the array,
*
* @return Unique_ptr with any data type (e.g std :: unique_ptr < float[] > x = std :: make_unique < float >(10);).
*
*/
template < typename type >
std :: unique_ptr < type > make_unique ( int size );

#endif

}

#endif // no std=c++17 support

#endif // __utils_h__
