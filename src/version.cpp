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

#include <version.h>

namespace info
{

std :: string get_version ()
{
  if ( __plasticity_major_version__ < 0 || __plasticity_minor_version__ < 0 || __plasticity_revision_version__ < 0 )
  {
    std :: cout << "Unknown.Unknown.Unknown" << std :: endl;
    std :: cerr << "Cannot deduce a correct version of plasticity! " << std :: endl
                << "Probably something goes wrong with the installation. " << std :: endl
                << "Please use the CMake file provided in the plasticity folder project to install the library as described in the project Docs. " << std :: endl
                << "Reference: https://github.com/Nico-Curti/plasticity"
                << std :: endl;

    return "Unknown.Unknown.Unknown";
  }

  std :: cout << "Plasticity version: "
              << __plasticity_major_version__    << "."
              << __plasticity_minor_version__    << "."
              << __plasticity_revision_version__ << std :: endl;

  return std :: to_string (__plasticity_major_version__) + "." + std :: to_string(__plasticity_minor_version__) + "." + std :: to_string(__plasticity_revision_version__);
}

bool get_viewer_support ()
{

#ifdef __viewer__

  std :: cout << "VIEWER support: ENABLE" << std :: endl;
  return true;

#else

  std :: cout << "VIEWER support: DISABLE" << std :: endl;
  return false;

#endif

}

} // end namespace
