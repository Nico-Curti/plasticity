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

#include <fmath.h>

namespace math
{

float pow2 (const float & x)
{

#ifdef __fast_math__

  const float offset = (x < 0   ) ? 1.f    : 0.f;
  const float clipp  = (x < -126) ? -126.f : x;
  const float z      = clipp - static_cast < int32_t >(clipp) + offset;
  union { u_int32_t i; float f; } v = {
    static_cast < u_int32_t >( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) )};

  return v.f;

#else

  return std :: pow(2.f, x);

#endif
}


float exp (const float & x)
{

#ifdef __fast_math__

  return math :: pow2(1.442695040f * x);

#else

  return std :: exp(x);

#endif
}


float log2 (const float & x)
{

#ifdef __fast_math__

  union { float f; u_int32_t i; } vx = { x };
  union { u_int32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
  float y = vx.i * 1.1920928955078125e-7f;

  return y - 124.22551499f - 1.498030302f * mx.f - 1.72587999f / (0.3520887068f + mx.f);

#else

  return std :: log2(x);

#endif
}


float log (const float & x)
{

#ifdef __fast_math__

  union { float x; u_int32_t i; } u { x };
  float y  = (u.i - magic_number) / (int_e - magic_number);
  float ey = math :: exp(y);
  y -= (ey - x) / ey;
  ey = math :: exp(y);
  y -= (ey - x) / ey;

  return y;

#else

  return std :: log(x);

#endif
}


float pow (const float & a, const float & b)
{

#ifdef __fast_math__

  return math :: pow2(b * math :: log2(a));

#else

  return std :: pow(a, b);

#endif
}


float log10 (const float & x)
{

#ifdef __fast_math__

  union { float x; u_int32_t i; } u { x };
  float y = (u.i - magic_number) / (int_10 - magic_number);
  float y10 = math :: pow(10, y);
  y -= (y10 - x)/( ln10 * y10 );
  y10 = math :: pow(10, y);
  y -= (y10 - x)/( ln10 * y10 );

  return y;

#else

  return std :: log10(x);

#endif
}


float atanh (const float & x)
{

#ifdef __fast_math__

  return .5f * math :: log((1.f + x) / (1.f - x));

#else

  return std :: atanh(x);

#endif
}


float tanh (const float & x)
{

#ifdef __fast_math__

  const float e = math :: pow2(- 2.88539008f * x); // math :: exp(-2.f * x);

  return (1.f - e) / (1.f + e);

#else

  return std :: tanh(x);

#endif
}


float hardtanh (const float & x)
{
  return (-1.f <= x && x <= 1.f) ? x : (x < -1.f) ? -1.f : 1.f;
}


float sqrt (const float & x)
{

#ifdef __fast_math__

  const float xhalf = 0.5f * x;
  float y;
  // get bits for floating value
  union { float x; int32_t i;} u;
  u.x = x;
  u.i = 0x5f3759df - ( u.i >> 1 );          // gives initial guess y0
  y   = u.x * ( 1.5f - xhalf * u.x * u.x ); // Newton step
  y   =   y * ( 1.5f - xhalf * y    * y  ); // second step to reach 1e-5 precision

  return x * y;

#else

  return std :: sqrt(x);

#endif
}


float rsqrt (const float & x)
{

#ifdef __fast_math__

  const float xhalf = x * 0.5f;
  float y;
  union { float x; long i;} u;
  u.x = x;
  u.i = 0x5f3759df - ( u.i >> 1 );          // what the fuck?
  y   = u.x * ( 1.5f - xhalf * u.x * u.x ); // 1st iteration
  y   = y   * ( 1.5f - xhalf * y   * y   ); // 2nd iteration, this can be removed

  return y;

#else

  return 1.f / std :: sqrt(x);

#endif
}



int32_t sign (const float & x)
{
  return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

int32_t gcd (const int32_t & a, const int32_t & b)  // greatest commond divisor
{
  return b == 0 ? a : gcd(b, a % b);
}

int32_t gd (const int32_t & a)  // greatest divisor
{
  if (!(a % 2))
    return a / 2;

  for (int32_t i = 3; i < a; i += 2)
    if (!(a % i))
      return a / i;

  return a;
}



} // end namespace
