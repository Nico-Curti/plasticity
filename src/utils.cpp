#include <utils.hpp>

namespace utils
{

  // Timing functions

  std :: chrono :: time_point < std :: chrono :: high_resolution_clock > what_time_is_it_now ()
  {
    return std :: chrono :: high_resolution_clock :: now ();
  }

  double elapsed_time (const std :: chrono :: time_point < std :: chrono :: high_resolution_clock > & start_time)
  {
    return static_cast < double > (std :: chrono :: duration_cast < std :: chrono :: milliseconds > (std :: chrono :: high_resolution_clock :: now () - start_time).count() ) * 1e-3; // seconds
  }

  void print_progress (const int & i, const int & num_iter, std :: chrono :: time_point < std :: chrono :: high_resolution_clock > & timer)
  {
    auto _it = utils :: elapsed_time(timer);
    timer = utils :: what_time_is_it_now();

    const float perc = static_cast < float >(i) / num_iter;

    int lpad = static_cast < int >(std :: floor(perc * PBWIDTH));
    lpad     = lpad > PBWIDTH ? PBWIDTH : lpad;
    int null_size = (PBWIDTH - 1 - lpad);
    null_size = null_size < 0 ? 0 : null_size;

    double c_it = i / _it;
    double m_it = (num_iter - i) / _it;

    c_it = std :: isnan(c_it) ? 0. : std :: isinf(c_it) ? 0. : c_it;
    m_it = std :: isnan(m_it) ? 0. : std :: isinf(m_it) ? 0. : m_it;

    std :: cout << RESET_COUT << "It: "
                << std :: right << i << " / "
                << std :: left  << num_iter
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


  // OS functions

  bool file_exists (const std :: string & filename)
  {
    // std :: filesystem :: exists (std :: filesystem :: path ( filename ) );
    if ( FILE * file = fopen(filename.c_str(), "r") )
    {
      fclose(file);
      return true;
    }

    return false;
  }

}
