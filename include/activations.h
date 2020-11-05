#ifndef __activations_h__
#define __activations_h__

#include <fmath.h>       // fast math functions

#include <unordered_map> // std :: unordered_map
#include <functional>    // std :: function

namespace transfer
{

  enum{ _logistic_ = 0, _loggy_, _relu_, _elu_, _relie_, _ramp_, _linear_, _tanh_, _plse_, _leaky_, _stair_, _hardtan_, _lhtan_, _selu_, _elliot_, _symm_elliot_, _softplus_, _softsign_, _asymm_logistic_
  };// activations

  static const std :: unordered_map < std :: string, int > get_activation {
                                                                            {"logistic"    , _logistic_},
                                                                            {"loggy"       , _loggy_},
                                                                            {"relu"        , _relu_},
                                                                            {"elu"         , _elu_},
                                                                            {"relie"       , _relie_},
                                                                            {"ramp"        , _ramp_},
                                                                            {"linear"      , _linear_},
                                                                            {"tanh"        , _tanh_},
                                                                            {"plse"        , _plse_},
                                                                            {"leaky"       , _leaky_},
                                                                            {"stair"       , _stair_},
                                                                            {"hardtan"     , _hardtan_},
                                                                            {"lhtan"       , _lhtan_},
                                                                            {"selu"        , _selu_},
                                                                            {"elliot"      , _elliot_},
                                                                            {"s_elliot"    , _symm_elliot_},
                                                                            {"softplus"    , _softplus_},
                                                                            {"softsign"    , _softsign_},
                                                                            {"as_logistic" , _asymm_logistic_}
                                                                          };

  static __unused float leaky_coeff = 1e-1f;
  static __unused float steepness = 1.f;

  float linear (const float & x);
  float g_linear (__unused const float & x);

  float stair (const float & x);
  float g_stair (const float & x);

  float hardtan (const float & x);
  float g_hardtan (const float & x);

  float logistic (const float & x);
  float g_logistic (const float & x);

  float loggy (const float & x);
  float g_loggy (const float & x);

  float relu (const float & x);
  float g_relu (const float & x);

  float elu (const float & x);
  float g_elu (const float & x);

  float relie (const float & x);
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

  std :: function < float(const float &) > activate ( const int & active);
  std :: function < float(const float &) > gradient ( const int & active);

}

#endif // __activations_h__
