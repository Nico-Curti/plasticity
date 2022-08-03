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

#include <activations.h>

namespace transfer
{
  static __unused float leaky_coeff = 1e-1f; ///< Internal coefficient of Leaky activation function
  static __unused float steepness = 1.f;     ///< Internal coefficient of Elliot function

  float linear (const float & x)
  {
    return x;
  }

  float g_linear (__unused const float & x)
  {
    return 1.f;
  }


  float stair (const float & x)
  {
    const int32_t n = static_cast < int32_t >(std :: floor (x));
    return (n % 2) ? (x - n) + std :: floor(x * .5f) : std :: floor(x * .5f);
  }

  float g_stair (const float & x)
  {
    return (std :: floor( x ) == x ) ? 0.f : 1.f;
  }


  float hardtan (const float & x)
  {
    return ( x < -2.5f ) ? 0.f : ( x > 2.5f ) ? 1.f : .2f * x + .5f;
  }

  float g_hardtan (const float & x)
  {
    return ( x > -2.5f && x < 2.5f ) ? .2f : 0.f;
  }


  float logistic (const float & x)
  {
    return 1.f / (1.f + math :: exp(- x));
  }

  float g_logistic (const float & x)
  {
    return (1.f - x) * x;
  }


  float loggy (const float & x)
  {
    return 2.f / (1.f + math :: exp(- x)) - 1.f;
  }

  float g_loggy (const float & x)
  {
    const float y = ( x + 1.f ) * .5f;
    return 2.f * (1.f - y) * y;
  }


  float relu (const float & x)
  {
    return ( x > 0.f ) ? x : 0.f;
  }

  float g_relu (const float & x)
  {
    return (x > 0.f) ? 1.f : 0.f;
  }


  float elu (const float & x)
  {
    return (x >= 0.f ) * x + ( x < 0.f ) * std :: expm1f(x);
  }

  float g_elu (const float & x)
  {
    return (x >= 0.f) + (x < 0.f) * (x + 1.f);
  }


  float relie (const float & x)
  {
    return (x > 0.f ) ? x : 1e-2f * x;
  }

  float g_relie (const float & x)
  {
    return (x > 0.f) ? 1.f : 1e-2f;
  }


  float ramp (const float & x)
  {
    return  x * ( x > 0.f ) + .1f * x;
  }

  float g_ramp (const float & x)
  {
    return (x > 0.f) + .1f;
  }


  float leaky (const float & x)
  {
    return (x > 0.f ) ? x : leaky_coeff * x;
  }

  float g_leaky (const float & x)
  {
    return (x > 0.f) ? 1.f : leaky_coeff;
  }


  float tanhy (const float & x)
  {
    return 2.f / (1.f + math :: exp (- (x + x))) - 1.f;
  }

  float g_tanhy (const float & x)
  {
    return 1.f - x * x;
  }


  float plse (const float & x)
  {
    return (x < -4.f ) ?
            1e-2f * ( x + 4.f ) :
            ( x > 4.f ) ?
            1e-2f * ( x - 4.f ) + 1.f :
            .125f * x + .5f;
  }

  float g_plse (const float & x)
  {
    return ( x < 0.f || x > 1.f) ? 1e-2f : .125f;
  }


  float lhtan (const float & x)
  {
    return x < 0.f ? 1e-3f * x : x > 1.f ? 1e-3f * (x - 1.f) + 1.f : x;
  }

  float g_lhtan (const float & x)
  {
    return ( x > 0.f && x < 1.f) ? 1.f : 1e-3f;
  }


  float selu (const float & x)
  {
    return (x >= 0.f) * 1.0507f * x + (x < 0.f) * 1.0507f * 1.6732f * std :: expm1f(x);
  }

  float g_selu (const float & x)
  {
    return (x >= 0.f) * 1.0507f + (x < 0.f) * (x + 1.0507f * 1.6732f);
  }

  float elliot (const float & x)
  {
    return .5f * steepness * x / (1.f + std :: fabs(x + steepness)) + .5f;
  }

  float g_elliot (const float & x)
  {
    const float last_forward = 1.f + std :: fabs(x * steepness); // miss null condition
    return .5f * steepness / (last_forward * last_forward);
  }

  float symm_elliot (const float & x)
  {
    return steepness * x / (1.f + std :: fabs(x * steepness));
  }

  float g_symm_elliot (const float & x)
  {
    const float last_forward = 1.f + std :: fabs(x * steepness); // miss null condition
    return steepness / (last_forward * last_forward);
  }

  float softplus (const float & x)
  {
    return std :: log1pf(math :: exp(x));
  }

  float g_softplus (const float & x)
  {
    const float ex = math :: exp(x);
    return ex / (1.f + ex);
  }

  float softsign (const float & x)
  {
    return x / (std :: fabs(x) + 1.f);
  }

  float g_softsign (const float & x)
  {
    const float fx = std :: fabs(x) + 1.f;
    return 1.f / (fx * fx);
  }

  float asymm_logistic (const float & x)
  {
    return x < 0.f ? -1.f * (2.f / (1.f + math :: exp(2.f * x)) - 1.f) :
                     50.f * (2.f / (1.f + math :: exp(-2.f * x / 50.f)) - 1.f);
  }

  float g_asymm_logistic (const float & x)
  {
    const float par = x < 0 ? -1.f : 50.f;
    //const float denom = 1.f + math :: exp(-2.f * x / par);
    //return 4.f * math :: exp(-2.f * x / par) / (denom * denom);
    const float temp = x / par;
    return (temp + 1.f) * (2.f - temp - 1.f);
  }

  std :: function < float(const float &) > activate ( const int32_t & active)
  {
    switch (active)
    {
      case transfer_t :: logistic:       return transfer :: logistic;
      case transfer_t :: loggy:          return transfer :: loggy;
      case transfer_t :: relu:           return transfer :: relu;
      case transfer_t :: elu:            return transfer :: elu;
      case transfer_t :: relie:          return transfer :: relie;
      case transfer_t :: ramp:           return transfer :: ramp;
      case transfer_t :: linear:         return transfer :: linear;
      case transfer_t :: Tanh:           return transfer :: tanhy;
      case transfer_t :: plse:           return transfer :: plse;
      case transfer_t :: leaky:          return transfer :: leaky;
      case transfer_t :: stair:          return transfer :: stair;
      case transfer_t :: hardtan:        return transfer :: hardtan;
      case transfer_t :: lhtan:          return transfer :: lhtan;
      case transfer_t :: selu:           return transfer :: selu;
      case transfer_t :: elliot:         return transfer :: elliot;
      case transfer_t :: symm_elliot:    return transfer :: symm_elliot;
      case transfer_t :: softplus:       return transfer :: softplus;
      case transfer_t :: softsign:       return transfer :: softsign;
      case transfer_t :: asymm_logistic: return transfer :: asymm_logistic;
      default:                           return nullptr;
    }
  }

  std :: function < float(const float &) > gradient ( const int32_t & active)
  {
    switch (active)
    {
      case transfer_t :: logistic:       return transfer :: g_logistic;
      case transfer_t :: loggy:          return transfer :: g_loggy;
      case transfer_t :: relu:           return transfer :: g_relu;
      case transfer_t :: elu:            return transfer :: g_elu;
      case transfer_t :: relie:          return transfer :: g_relie;
      case transfer_t :: ramp:           return transfer :: g_ramp;
      case transfer_t :: linear:         return transfer :: g_linear;
      case transfer_t :: Tanh:           return transfer :: g_tanhy;
      case transfer_t :: plse:           return transfer :: g_plse;
      case transfer_t :: leaky:          return transfer :: g_leaky;
      case transfer_t :: stair:          return transfer :: g_stair;
      case transfer_t :: hardtan:        return transfer :: g_hardtan;
      case transfer_t :: lhtan:          return transfer :: g_lhtan;
      case transfer_t :: selu:           return transfer :: g_selu;
      case transfer_t :: elliot:         return transfer :: g_elliot;
      case transfer_t :: symm_elliot:    return transfer :: g_symm_elliot;
      case transfer_t :: softplus:       return transfer :: g_softplus;
      case transfer_t :: softsign:       return transfer :: g_softsign;
      case transfer_t :: asymm_logistic: return transfer :: g_asymm_logistic;
      default:                           return nullptr;
    }
  }


}
