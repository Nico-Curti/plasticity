#ifndef __activations_h__
#define __activations_h__

#include <fmath.h>       // fast math functions

#include <unordered_map> // std :: unordered_map
#include <functional>    // std :: function


enum transfer_t { logistic = 0, loggy, relu, elu, relie, ramp, linear, Tanh, plse, leaky, stair, hardtan, lhtan, selu, elliot, symm_elliot, softplus, softsign, asymm_logistic, sigmoid = logistic
}; ///< activation types


namespace transfer
{


  static const std :: unordered_map < std :: string, int > get_activation {
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
  * ```python
  * f(x) = x
  * ```
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
  * ```python
  * f'(x) = 1
  * ```
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
  * ```python
  * f(x) = ...
  * ```
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
  * ```python
  * f'(x) = ...
  * ```
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
  * ```python
  * if x < -2.5:
  *   return 0.
  * elif x > 2.5:
  *   return 1.
  * else:
  *   retun 0.2 * x + 0.5
  * ```
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
  * ```python
  * if x > -2.5 and x < 2.5:
  *   return 0.2
  * else:
  *   return 0.0
  * ```
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
  * ```python
  * f(x) = 1. / (1. + exp(-x))
  * ```
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
  * ```python
  * f'(x) = (1. - x) * x
  * ```
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
  * ```python
  * f(x) = 2. / (1. + exp(-x)) - 1.
  * ```
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
  * ```python
  * y = (x + 1.) * 0.5
  * f'(x) = 2. * (1. - y) * y
  * ```
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
  * ```python
  * f(x) = max(0, x)
  * ```
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
  * ```python
  * f'(x) = 1 if x > 0 else 0
  * ```
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
  * ```python
  * y = x >= 0
  * f(x) = y * x + ~y * exp(x - 1.)
  * ```
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
  * ```python
  * y = x >= 0
  * f'(x) = y + ~y * (x + 1.)
  * ```
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
  * ```python
  * f(x) = x if x > 0 else 0.001 * x
  * ```
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
  * ```python
  * f'(x) = 1 if x > 0 else 0.001
  * ```
  *
  * @param x Input variable.
  *
  * @return The gradient of the input.
  */
  float g_relie (const float & x);

  float ramp (const float & x);
  float g_ramp (const float & x);

  float leaky (const float & x);
  float g_leaky (const float & x);

  float tanhy (const float & x);
  float g_tanhy (const float & x);

  float plse (const float & x);
  float g_plse (const float & x);

  float lhtan (const float & x);
  float g_lhtan (const float & x);

  float selu (const float & x);
  float g_selu (const float & x);

  float elliot (const float & x);
  float g_elliot (const float & x);

  float symm_elliot (const float & x);
  float g_symm_elliot (const float & x);

  float softplus (const float & x);
  float g_softplus (const float & x);

  float softsign (const float & x);
  float g_softsign (const float & x);

  float asymm_logistic (const float & x);
  float g_asymm_logistic (const float & x);

  void swish_array (const float * x, const int & n, float * output_sigmoid, float * output);
  void swish_gradient (const float * x, const int & n, const float * sigmoid, float * delta);

  void mish_array (const float * x, const int & n, float * input_activation, float * output);
  void mish_gradient (const int & n, const float * activation_input, float * delta);

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
  std :: function < float(const float &) > activate ( const int & active );
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
  std :: function < float(const float &) > gradient ( const int & active );

}

#endif // __activations_h__
