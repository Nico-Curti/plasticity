#ifndef __utils_hpp__
#define __utils_hpp__

#include <utils.h>

namespace utils
{


template < typename Time >
double elapsed_time (const Time & start_time)
{
#ifdef _OPENMP

  return omp_get_wtime() - start_time;

#elif defined _MPI

  return start_time.elapsed();

#else

  return static_cast < double > (std :: chrono :: duration_cast < std :: chrono :: milliseconds > (std :: chrono :: high_resolution_clock :: now () - start_time).count() ) * 1e-3; // seconds

#endif
}


template < typename Time >
void print_progress (const int & i, const int & num_batches, Time & timer)
{
  const float perc = static_cast < float >(i) / num_batches;

  int lpad = static_cast < int >(std :: floor(perc * PBWIDTH));
  lpad     = lpad > PBWIDTH ? PBWIDTH : lpad;
  int null_size = (PBWIDTH - 1 - lpad);
  null_size = null_size < 0 ? 0 : null_size;

  auto _it = utils :: elapsed_time(timer);
  timer = utils :: what_time_is_it_now();

  double c_it = i / _it;
  double m_it = (num_batches - i) / _it;

  c_it = std :: isnan(c_it) ? 0. : std :: isinf(c_it) ? 0. : c_it;
  m_it = std :: isnan(m_it) ? 0. : std :: isinf(m_it) ? 0. : m_it;

  std :: cout << RESET_COUT << "It: "
              << std :: right << i << " / "
              << std :: left  << num_batches
              << " |"
              << std :: setw(PBWIDTH)
              << FILL_VALUE * lpad + NULL_VALUE * null_size
              << "|  ["
              << std :: right << c_it << ":00"
              << "<"
              << std :: right << m_it << ":00"
              << " sec, "
              << std :: right << 1.f / _it << " it/sec]"
              ;
  std :: cout << std :: flush;
}

} // end namespace utils


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



#if __cplusplus < 201700 && !defined _MSC_VER // no std=c++17 support

namespace std
{


template < class type >
const type & clamp ( const type & v, const type & lo, const type & hi )
{
#ifdef DEBUG

  assert( ! (hi < lo) );

#endif

  return v < lo ? lo : hi > v ? hi : v;
}

#if ( ( __cplusplus < 201100 && !(_MSC_VER) ) || ( __GNUC__ == 4 && __GNUC_MINOR__ < 9) && !(__clang__) )

template < typename type >
std :: unique_ptr < type > make_unique ( int size )
{
  return std :: unique_ptr < type > ( new typename std :: remove_extent < type > :: type[size] () );
}

#endif

}

#endif // no std=c++17 support


#endif // __utils_hpp__
