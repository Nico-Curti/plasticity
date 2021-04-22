#include <version.h>

void print_help ()
{
  std :: cerr << "Usage: ./plasticity_infos [--version] [--viewer]" << std :: endl
              << "plasticity installation infos function" << std :: endl
              << std :: endl
              << "optional arguments:" << std :: endl
              << "        --version                    Get current plasticity installation version"              << std :: endl
              << "        --viewer                     Check if current plasticity installation supports Viewer" << std :: endl
              << std :: endl;
  std :: exit(0);
}


void parse_infos (int argc, char ** argv,
                  bool & version,
                  bool & viewer_support
                  )
{
  for (int i = 1; i < argc; ++i)
  {
    std :: string arg(argv[i]);

    if ( arg == "--help" )
      print_help();

    else if ( arg == "--version" )
      version = true;

    else if ( arg == "--viewer" )
      viewer_support = true;

    else
      continue;
  }


  if ( ! ( version || viewer_support ) )
  {
    std :: cerr << "Invalid usage! Use the --help or -h commands to see the list of available arguments" << std :: endl;
    std :: exit(1);
  }

  return;
}



int main (int argc, char ** argv)
{
  bool version = false;
  bool viewer = false;

  /******************************************************
                      Parse Command Line
  *******************************************************/

  parse_infos (argc, argv, version, viewer);

  if ( version )
    info :: get_version ();

  if ( viewer )
    info :: get_viewer_support ();

  return 0;

}
