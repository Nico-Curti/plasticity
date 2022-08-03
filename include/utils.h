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
#include <iterator>  // std :: begin

#include <fmath.h>   // fast math function
#include <chrono>    // timing function

#ifdef DEBUG

  #include <cassert>

#endif

// Usefull macro

/// @cond DEF
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
/// @endcond

// Usefull variables

#define PBWIDTH 20                           ///< Width of the progress bar
static const std :: string FILL_VALUE = "█"; ///< fill value for the progress bar
static const std :: string NULL_VALUE = " "; ///< empty value for the progress bar

#ifdef _WIN32
  #define RESET_COUT '\r'       ///< Return carriage value in Window OS
#else
  #define RESET_COUT "\r\x1B[K" ///< Return carriage value in Unix OS
#endif


// Usefull common variables

namespace utils
{

  // Timing functions

  /**
  * @brief Return the current time (STD format).
  *
  * @details There are many specialization of this function according to different parallel environments.
  *
  * @return The current time as chrono timer value.
  *
  */
  std :: chrono :: time_point < std :: chrono :: high_resolution_clock > what_time_is_it_now ();

  /**
  * @brief Get the elapsed time from the start.
  *
  * @details This function is related to the what_time_is_it_now function and it returns
  * the elapsed time as seconds. For an accurate computation the time evaluation is performed
  * using milliseconds as template.
  *
  * @param start_time Starting time.
  *
  * @return The elapsed time as double.
  *
  */
  double elapsed_time (const std :: chrono :: time_point < std :: chrono :: high_resolution_clock > & start_time);

  /**
  * @brief Print progress bar
  *
  * @details This function is used to progressively print a progress bar
  * with timer.
  *
  * @param i Current iteration value.
  * @param num_iter Maximum number of iterations, i.e the width of the progress bar.
  * @param timer Start time (the timer is reset at the end of the function).
  *
  */
  void print_progress (const int & i, const int & num_iter,
    std :: chrono :: time_point < std :: chrono :: high_resolution_clock > & timer);

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

} // end namespace utils


/**
* @brief Overload operator * between strings and integers
*
* @details This overload allows to multiply and thus replicate a string
* for a fixed number of times.
*
* @tparam Char string data type
* @tparam Traits Second general-string template type
* @tparam Allocator Third general-string template type
* @param s Input string to multiply.
* @param n Number of times to repeat the string.
*
* @return A string repeated n times.
*
*/
template < typename Char, typename Traits, typename Allocator >
std :: basic_string < Char, Traits, Allocator > operator * (
  const std :: basic_string < Char, Traits, Allocator > & s, std :: size_t n);

/**
* @brief Overload operator * between integers and strings
*
* @details This overload allows to multiply and thus replicate a string
* for a fixed number of times.
*
* @tparam Char string data type
* @tparam Traits Second general-string template type
* @tparam Allocator Third general-string template type
* @param n Number of times to repeat the string.
* @param s Input string to multiply.
*
* @return A string repeated n times.
*
*/
template < typename Char, typename Traits, typename Allocator >
std :: basic_string < Char, Traits, Allocator > operator * (
  std :: size_t n, const std :: basic_string < Char, Traits, Allocator > & s);

#endif // __utils_h__
