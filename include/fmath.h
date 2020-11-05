#ifndef __fmath_h__
#define __fmath_h__

#include <utils.h>  // common math variables

#include <cmath>

namespace math
{

static const float int_e        = 1076754516.f;
static const float int_10       = 1092616192.f;
static const float log2_over_2  = 0.34657359027997264311f;

#if defined _MSC_VER || defined __clang__

  static const float magic_number = 1064992212.25472f;
  static const float ln10         = 2.302585092994046f;

#else

  static constexpr float magic_number = std :: pow(2.f, 23) * (127.f - 0.043035);
  static constexpr float ln10         = std :: log(10.f);

#endif


float pow2 (const float & p);
float exp (const float & p);
float log2 (const float & x);
float log (const float & x);
float pow (const float & a, const float & b);
float log10 (const float & x);
float atanh (const float & x);
float tanh (const float & x);
float hardtanh (const float & x);
float sqrt (const float & x);
float rsqrt (const float & x);

int sign (const float & x);
int gcd (const int & a, const int & b); // greatest commond divisor
int gd (const int & a); // greatest divisor
}

#endif // __fmath_h__
