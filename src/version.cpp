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
