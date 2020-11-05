#include <utils.hpp>

namespace utils
{

  // Timing functions

#if defined _OPENMP

  double what_time_is_it_now ()
  {
    return omp_get_wtime();
  }

#elif defined _MPI

  boost :: mpi :: timer what_time_is_it_now ()
  {
    return boost :: mpi :: timer();
  }

#else

  std :: chrono :: time_point < std :: chrono :: high_resolution_clock > what_time_is_it_now ()
  {
    return std :: chrono :: high_resolution_clock :: now ();
  }

#endif


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
