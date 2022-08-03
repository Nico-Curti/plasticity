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

#ifndef __fmath_h__
#define __fmath_h__

#include <utils.h>  // common math variables

#include <cmath>    // std :: math functions

namespace math
{

static const float int_e        = 1076754516.f;            ///< integer representation of Euler number
static const float int_10       = 1092616192.f;            ///< integer representation of 10.f
static const float log2_over_2  = 0.34657359027997264311f; ///< std :: log(2) / 2 as float

#if defined _MSC_VER || defined __clang__

  static const float magic_number = 1064992212.25472f;     ///< std :: pow(2.f, 23) * (127.f - 0.043035);
  static const float ln10         = 2.302585092994046f;    ///< std :: log(10) as float

#else

  static constexpr float magic_number = std :: pow(2.f, 23) * (127.f - 0.043035); ///< std :: pow(2.f, 23) * (127.f - 0.043035);
  static constexpr float ln10         = std :: log(10.f);                         ///< std :: log(10) as float

#endif

/**
* @brief Fast approximation of power of 2.
*
* @details The approximation is given by a reinterpretation of binary form of the float variables
*
* @note The fast implementation is available if __fast_math__ is defined at compile time.
* Otherwise the evaluation is performed using std functions.
*
* @param p Input variable.
*
* @return The result of the operation.
*/
float pow2 (const float & p);

/**
* @brief Fast approximation of exp function.
*
* @details The approximation is given by a reinterpretation of binary form of the float variables
*
* @note The fast implementation is available if __fast_math__ is defined at compile time.
* Otherwise the evaluation is performed using std functions.
*
* @param p Input variable.
*
* @return The result of the operation.
*/
float exp (const float & p);

/**
* @brief Fast approximation of log2 function.
*
* @details The approximation is given by a reinterpretation of binary form of the float variables
*
* @note The fast implementation is available if __fast_math__ is defined at compile time.
* Otherwise the evaluation is performed using std functions.
*
* @param x Input variable.
*
* @return The result of the operation.
*/
float log2 (const float & x);

/**
* @brief Fast approximation of log function.
*
* @details The approximation is given by a reinterpretation of binary form of the float variables
*
* @note The fast implementation is available if __fast_math__ is defined at compile time.
* Otherwise the evaluation is performed using std functions.
*
* @param x Input variable.
*
* @return The result of the operation.
*/
float log (const float & x);

/**
* @brief Fast approximation of pow function.
*
* @details The approximation is given by a reinterpretation of binary form of the float variables
*
* @note The fast implementation is available if __fast_math__ is defined at compile time.
* Otherwise the evaluation is performed using std functions.
*
* @param a Input variable (base).
* @param b Input variable (exponent).
*
* @return The result of the operation.
*/
float pow (const float & a, const float & b);

/**
* @brief Fast approximation of log10 function.
*
* @details The approximation is given by a reinterpretation of binary form of the float variables
*
* @note The fast implementation is available if __fast_math__ is defined at compile time.
* Otherwise the evaluation is performed using std functions.
*
* @param x Input variable.
*
* @return The result of the operation.
*/
float log10 (const float & x);

/**
* @brief Fast approximation of atanh function.
*
* @details The approximation is given by a simplified version of the math formula.
*
* @note The fast implementation is available if __fast_math__ is defined at compile time.
* Otherwise the evaluation is performed using std functions.
*
* @param x Input variable.
*
* @return The result of the operation.
*/
float atanh (const float & x);

/**
* @brief Fast approximation of tanh function.
*
* @details The approximation is given by a simplified version of the math formula.
*
* @note The fast implementation is available if __fast_math__ is defined at compile time.
* Otherwise the evaluation is performed using std functions.
*
* @param x Input variable.
*
* @return The result of the operation.
*/
float tanh (const float & x);

/**
* @brief Fast approximation of hardtanh function.
*
* @details The approximation is given by a reinterpretation of binary form of the float variables
*
* @note The fast implementation is available if __fast_math__ is defined at compile time.
* Otherwise the evaluation is performed using std functions.
*
* @param x Input variable.
*
* @return The result of the operation.
*/
float hardtanh (const float & x);

/**
* @brief Fast approximation of sqrt function.
*
* @details The Hard Tanh is evaluated as:
*
* .. code-block:: python
*
*   if x < -1:
*     return -1.
*   elif x <= -1 and x <= 1.:
*     return x
*   else:
*     return 1.
*
* @param x Input variable.
*
* @return The result of the operation.
*/
float sqrt (const float & x);

/**
* @brief Fast approximation of reverse sqrt function.
*
* @details The approximation is given by Newton strategy.
*
* @note The fast implementation is available if __fast_math__ is defined at compile time.
* Otherwise the evaluation is performed using std functions.
*
* @param x Input variable.
*
* @return The result of the operation.
*/
float rsqrt (const float & x);

/**
* @brief Evaluate the sign of the variable.
*
* @details It is just a check of the variable. If it is zero the sign is always zero.
*
* @param x Input variable.
*
* @return The sign of the input as [-1, 0, 1].
*/
int32_t sign (const float & x);

/**
* @brief Gratest Common Divisor between the two variables.
*
* @details The result is obtained by a recursive call of this function.
*
* @param a Input variable.
* @param b Input variable.
*
* @return The GCD of the two inputs.
*/
int32_t gcd (const int32_t & a, const int32_t & b); // greatest commond divisor

/**
* @brief Greatest divisor (GD) of the number.
*
* @details The GD is found by an iteratively division of the number.
*
* @param a Input variable.
*
* @return The greatest divisor of the input
*/
int32_t gd (const int32_t & a);
}

#endif // __fmath_h__
