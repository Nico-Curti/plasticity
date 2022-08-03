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
