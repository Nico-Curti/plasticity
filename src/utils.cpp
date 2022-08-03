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

  void print_progress (const int32_t & i, const int32_t & num_iter, std :: chrono :: time_point < std :: chrono :: high_resolution_clock > & timer)
  {
    auto _it = utils :: elapsed_time(timer);
    timer = utils :: what_time_is_it_now();

    const float perc = static_cast < float >(i) / num_iter;

    int32_t lpad = static_cast < int32_t >(std :: floor(perc * PBWIDTH));
    lpad     = lpad > PBWIDTH ? PBWIDTH : lpad;
    int32_t null_size = (PBWIDTH - 1 - lpad);
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
