#ifndef __utils_hpp__
#define __utils_hpp__

#include <utils.h>

template < typename Char, typename Traits, typename Allocator >
std :: basic_string < Char, Traits, Allocator > operator * (const std :: basic_string < Char, Traits, Allocator > & s, std :: size_t n)
{
  std :: basic_string < Char, Traits, Allocator > tmp;
  for (std :: size_t i = 0; i < n; ++i) tmp += s;
  return tmp;
}

template < typename Char, typename Traits, typename Allocator >
std :: basic_string < Char, Traits, Allocator > operator * (std :: size_t n, const std :: basic_string < Char, Traits, Allocator > & s)
{
  return s * n;
}


#endif // __utils_hpp__
