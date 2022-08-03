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

#ifndef __activations_h__
#define __activations_h__

#include <fmath.h>       // fast math functions

#include <unordered_map> // std :: unordered_map
#include <functional>    // std :: function


enum transfer_t { logistic = 0, loggy, relu,
                  elu, relie, ramp,
                  linear, Tanh, plse,
                  leaky, stair, hardtan,
                  lhtan, selu, elliot,
                  symm_elliot, softplus,
                  softsign, asymm_logistic,
                  sigmoid = logistic
}; ///< activation types


namespace transfer
{


  static const std :: unordered_map < std :: string, int32_t > get_activation {
    {"logistic"    , logistic},
    {"sigmoid"     , logistic},
    {"loggy"       , loggy},
    {"relu"        , relu},
    {"elu"         , elu},
    {"relie"       , relie},
    {"ramp"        , ramp},
    {"linear"      , linear},
    {"tanh"        , Tanh},
    {"plse"        , plse},
    {"leaky"       , leaky},
    {"stair"       , stair},
    {"hardtan"     , hardtan},
    {"lhtan"       , lhtan},
    {"selu"        , selu},
    {"elliot"      , elliot},
    {"s_elliot"    , symm_elliot},
    {"softplus"    , softplus},
    {"softsign"    , softsign},
    {"as_logistic" , asymm_logistic}
  }; ///< Utility for the activations management

  /**
  * @brief Linear activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = x
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float linear (const float & x);
  /**
  * @brief Gradient of the Linear activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = 1
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_linear (const float & x);

  /**
  * @brief Stair activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = (x - floor(x)) + floor(x * 0.5) if (floor(x) % 2) else floor(x * 0.5)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float stair (const float & x);
  /**
  * @brief Gradient of the Stair activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = 0 if (floor(x) == x) else 1
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_stair (const float & x);

  /**
  * @brief HardTan activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = 0 if x < 2.5 else (1 if x > 2.5 else 0.2 * x + 0.5)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float hardtan (const float & x);
  /**
  * @brief Gradient of the HardTan activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = 0.2 if x > -2.5 and x < 2.5 else 0
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_hardtan (const float & x);

  /**
  * @brief Logistic (sigmoid) activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = 1 / (1 + \exp(-x))
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float logistic (const float & x);
  /**
  * @brief Gradient of the Logistic activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = (1 - x) * x
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_logistic (const float & x);

  /**
  * @brief Loggy activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = 2 / (1 + \exp(-x)) - 1
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float loggy (const float & x);
  /**
  * @brief Gradient of the Loggy activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * y = (x + 1.) * 0.5
  * f'(x) = 2. * (1. - y) * y
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_loggy (const float & x);

  /**
  * @brief ReLU activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = max(0, x)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float relu (const float & x);
  /**
  * @brief Gradient of the ReLU activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = 1 if x > 0 else 0
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_relu (const float & x);

  /**
  * @brief Elu activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * y = x >= 0
  * f(x) = y * x + ~y * exp(x - 1.)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float elu (const float & x);
  /**
  * @brief Gradient of the Elu activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * y = x >= 0
  * f'(x) = y + ~y * (x + 1.)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_elu (const float & x);

  /**
  * @brief Relie activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = x if x > 0 else 0.001 * x
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float relie (const float & x);
  /**
  * @brief Gradient of the Relie activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = 1 if x > 0 else 0.001
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_relie (const float & x);

  /**
  * @brief Ramp activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = x * ( x > 0 ) + 0.1 * x
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float ramp (const float & x);
  /**
  * @brief Gradient of the Ramp activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = (x > 0) + 0.1
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_ramp (const float & x);

  /**
  * @brief Leaky activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = x if (x > 0 ) else leaky_coeff * x
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float leaky (const float & x);
  /**
  * @brief Gradient of the Leaky activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = 1 if x > 0 else leaky_coeff
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_leaky (const float & x);

  /**
  * @brief Tanh activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = 2 / (1 + \exp(- (x + x))) - 1
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float tanhy (const float & x);
  /**
  * @brief Gradient of the Tanh activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = 1 - x^2
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_tanhy (const float & x);

  /**
  * @brief Plse activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = 0.01 * ( x + 4 ) if (x < -4 ) else (0.01 * (x - 4) + 1 if (x > 4) else 0.125 * x + 0.5)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float plse (const float & x);
  /**
  * @brief Gradient of the Plse activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = 0.01 if (x < 0 | x > 1) else 0.125
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_plse (const float & x);

  /**
  * @brief LhTan activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = 0.001 * x if x < 0 else (0.001 * (x - 1) + 1 if x > 1 else x)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float lhtan (const float & x);
  /**
  * @brief Gradient of the LhTan activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = 1 if ( x > 0 & x < 1 ) else 0.001
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_lhtan (const float & x);

  /**
  * @brief Selu activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = (x \geq 0) * 1.0507 * x + (x < 0) * 1.0507 * 1.6732 * expm1(x)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float selu (const float & x);
  /**
  * @brief Gradient of the Selu activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = (x \geq 0) * 1.0507 + (x < 0) * (x + 1.0507 * 1.6732)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_selu (const float & x);

  /**
  * @brief Elliot activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = 0.5 * steepness * x / (1 + |x + steepness|) + 0.5
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float elliot (const float & x);
  /**
  * @brief Gradient of the Elliot activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * last_forward = 1 + |x * steepness|
  * f'(x) = 0.5 * steepness / (last_forward * last_forward)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_elliot (const float & x);

  /**
  * @brief Symmetric Elliot activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = steepness * x / (1 + |x + steepness|)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float symm_elliot (const float & x);
  /**
  * @brief Gradient of the Symmetric Elliot activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * last_forward = 1 + |x * steepness|
  * f'(x) = steepness / (last_forward * last_forward)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_symm_elliot (const float & x);

  /**
  * @brief SoftPlus activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = log1pf(\exp(x))
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float softplus (const float & x);
  /**
  * @brief Gradient of the SoftPlus activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = \exp(x) / (1 + \exp(x))
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_softplus (const float & x);

  /**
  * @brief SoftSign activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = x / (|x| + 1)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float softsign (const float & x);
  /**
  * @brief Gradient of the SoftSign activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * f'(x) = 1 / (|x| + 1)^2
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_softsign (const float & x);

  /**
  * @brief Asymmetric Logistic activation function.
  *
  * @details The activation function follows the equation:
  *
  * \f[
  * f(x) = -1 * (2 / (1 + \exp(2 * x)) - 1) if x < 0 else 50 * (2 / (1 + \exp(-2 * x / 50)) - 1)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The activated input.
  */
  float asymm_logistic (const float & x);
  /**
  * @brief Gradient of the Asymmetric Logistic activation function.
  *
  * @details The gradient is equal to:
  *
  * \f[
  * par = -1 if x < 0 else 50
  * f'(x) = (x / par + 1) * (2 - x / par - 1)
  * \f]
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_asymm_logistic (const float & x);

  /**
  * @brief Switch case between activation functions.
  *
  * @details This function is used to set the desired activation function
  * (returned as pointer to function) starting from its "name" in the enum.
  * If the input integer is not in the enum range a nullptr is returned.
  *
  * @param active Integer from the enum activation types.
  *
  * @return Pointer to the desired function.
  */
  std :: function < float(const float &) > activate ( const int32_t & active );
  /**
  * @brief Switch case between gradient functions.
  *
  * @details This function is used to set the desired gradient function
  * (returned as pointer to function) starting from its "name" in the enum.
  * If the input integer is not in the enum range a nullptr is returned.
  *
  * @param active Integer from the enum activation types.
  *
  * @return Pointer to the desired function.
  */
  std :: function < float(const float &) > gradient ( const int32_t & active );

}

#endif // __activations_h__
