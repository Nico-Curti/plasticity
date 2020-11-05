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
              << FILL_VALUE * lpad + NULL_VALUE * (PBWIDTH - 1 - lpad)
              << "|  ["
              << std :: right << c_it << ":00"
              << "<"
              << std :: right << m_it << ":00"
              << " sec, "
              << std :: right << 1.f / _it << " it/sec]"
              ;
  std :: cout << std :: flush;
}


template < typename Char, typename Traits, typename Allocator >
std :: basic_string < Char, Traits, Allocator > operator * (const std :: basic_string < Char, Traits, Allocator > & s, std :: size_t n)
{
  std :: basic_string < Char, Traits, Allocator > tmp = s;
  for (std :: size_t i = 0; i < n; ++i) tmp += s;
  return tmp;
}

template < typename Char, typename Traits, typename Allocator >
std :: basic_string < Char, Traits, Allocator > operator * (std :: size_t n, const std :: basic_string < Char, Traits, Allocator > & s)
{
  return s * n;
}


#ifdef _OPENMP

#define __minimum_sort_size__ 1000

// parallel merge argsort

template < typename Order >
void mergeargsort_serial ( std :: pair < float, int > * array, const float * a, const int & start, const int & end, Order ord)
{
  if ( (end - start) == 2 )
  {
    if ( ord(array[start], array[end - 1]) )
      return;
    else
    {
      std :: swap(array[start], array[end - 1]);
      return;
    }
  }

  const int pivot = start + ((end - start) >> 1);

  if ((end - start) < __minimum_sort_size__)
    std :: sort(array + start, array + end, ord);
  else
  {
    mergeargsort_serial(array, a, start, pivot, ord);
    mergeargsort_serial(array, a, pivot, end, ord);
  }

  std :: inplace_merge(array + start, array + pivot, array + end, ord);

  return;
}

template < typename Order >
void mergeargsort_parallel_omp ( std :: pair < float, int > * array, const float * a, const int & start, const int & end, const int & threads, Order ord)
{
  const int pivot = start + ((end - start) >> 1);

  if (threads <= 1)
  {
    mergeargsort_serial(array, a, start, end, ord);
    return;
  }
  else
  {
#pragma omp task shared(start, end, threads)
    {
      mergeargsort_parallel_omp(array, a, start, pivot, threads >> 1, ord);
    }
#pragma omp task shared(start, end, threads)
    {
      mergeargsort_parallel_omp(array, a, pivot, end, threads - (threads >> 1), ord);
    }
#pragma omp taskwait
  }

  std :: inplace_merge(array + start, array + pivot, array + end, ord);

  return;
}

template < typename Order >
void argsort (const float * a, int * indexes, const int & start, const int & end, Order ord)
{
  // TODO: better testing!
  static std :: unique_ptr < std :: pair < float, int > [] > array;

  #pragma omp single
  array = std :: make_unique < std :: pair < float, int > [] >(end - start);

  #pragma omp for
  for (int i = 0; i < end - start; ++i)
    array[i] = std :: make_pair(a[i + start], i + start);

  const int nth = (omp_get_num_threads() % 2) ? omp_get_num_threads() - 1 : omp_get_num_threads();
  const int diff = end % nth;
  const int size = diff ? end - diff : end;

  #pragma omp single
  #pragma omp taskgroup
  {

    mergeargsort_parallel_omp(array.get(), a, start, size, nth, ord);

  } // end single section

  if (diff)
  {
    std :: sort(array.get() + size, array.get() + end, ord);
    std :: inplace_merge(array.get() + start, array.get() + size, array.get() + end, ord);
  }

  #pragma omp for
  for (int i = 0; i < end - start; ++i)
    indexes[i] = std :: get < 1 >(array[i]);
}

#endif

} // end namespace utils


#if __cplusplus < 201700 && !defined _MSC_VER // no std=c++17 support

namespace std
{

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
