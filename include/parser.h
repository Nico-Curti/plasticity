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

#ifndef __parser_h__
#define __parser_h__

#include <unordered_map> // std :: unordered_map
#include <sstream>       // std :: stringstream
#include <fstream>       // std :: ifstream
#include <regex>         // std :: regex
#include <iterator>      // std :: ostream_iterator
#include <type_traits>   // std :: false

#include <utils.hpp>     // utility functions

namespace parser
{


/**
* @class config
*
* @brief Config file parser
*
* @details This class implements a simplified version of an INI
* config parser. Each line of the file must be in the format "keyword=value".
* The data table is stored internally as string-string lut.
* The conversion to the required data type is obtained using the get member function.
*
*/
class config
{

  // Private members

  static std :: regex comment_regex; ///< regex comment, aka all tokens after a "#" character
  static std :: regex value_regex; ///< regex for the value split

  std :: unordered_map < std :: string, std :: string > map; ///< internal data lut

public:

  // Constructors

  /**
  * @brief Read and parse the config file.
  *
  * @details The constructor read and parse the provided config file.
  * If the file doesn't exist a runtime_error is throwed.
  * The config file can includes comment with the escape character "#".
  * Available configuration variables are only [int, float, string].
  * Class lut can help to convert string to appropriated variable instances (es. activation).
  *
  * @note The object can be instanced only with a valid filename.
  *
  * @param filename Input configuration filename/path.
  *
  */
  explicit config (const std :: string & filename);

  // Destructors

  /**
  * @brief Destructor.
  *
  * @details Completely delete the object and release the memory of the arrays.
  *
  */
  ~config () = default;

  // Public member functions

  /**
  * @brief Getter.
  *
  * @details Getter function of the parser. The returning value type
  * must be provided by the user.
  *
  * @param key String name of the variable.
  * @param def Default value of the variable if not present in the LUT.
  *
  * @tparam type Data type required by the user.
  *
  */
  template < typename type >
  type get (const std :: string & key, const type & def) const;

private:

  // Private member functions

  /**
  * @brief Parse the INI format
  *
  * @details Core function of the config parser. Inside this function the
  * possible behaviors are managed using the static regex variables.
  * At the end the internal LUT is filled with the correct values.
  *
  * @param buffer String buffer of the input config file.
  *
  */
  void parse_ini ( std :: stringstream & buffer );

};

} // end namespace

#endif // __parser_h__
