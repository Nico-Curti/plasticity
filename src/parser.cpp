#include <parser.h>


namespace parser
{

std :: regex config :: comment_regex = std :: regex (R"x(\s*[;#])x");

#ifdef _MSC_VER // the other (correct) regex is too complex for Visual Studio (-.-)

  std :: regex config :: value_regex = std :: regex (R"x(\s*([^ \t=]*)\s*[=:]\s*([^&]*)&?)x");

#else

  std :: regex config :: value_regex = std :: regex (R"x(\s*(\S[^ \t=]*)\s*[=:]\s*(((\s?\S+)\s+*)+)\s*$)x");

#endif


// Data Config functions

config :: config (const std :: string & filename)
{
  if ( ! utils :: file_exists(filename) )
    throw std :: runtime_error("Config file not found. Given: " + filename);

  std :: ifstream in(filename);

  std :: stringstream buffer;
  buffer << in.rdbuf();
  in.close();

  // parse ini file fmt
  this->parse_ini(buffer);
}

void config :: parse_ini (std :: stringstream & buffer)
{
  std :: string current_section;
  std :: smatch pieces;

  for (std :: string line; std :: getline(buffer, line); )
  {
    // (key, values) match
    if (std :: regex_match(line, pieces, config :: value_regex))
    {
      if (pieces.size() == 5) // exactly enough matches
        this->map[pieces[1].str()] = pieces[2].str();
    }
    // skip comment lines and blank lines
    else if (line.empty() || std :: regex_match(line, pieces, config :: comment_regex))
      continue;
  }
}

// Data Config functions

template < >
int config :: get < int > (const std :: string & key, const int & def) const
{
  return (this->map.find(key) != this->map.end()) ?
          std :: stoi(this->map.at(key))          :
          def;
}

template < >
float config :: get < float > (const std :: string & key, const float & def) const
{
  return (this->map.find(key) != this->map.end()) ?
          std :: stof(this->map.at(key))          :
          def;
}

template < >
std :: string config :: get < std :: string > (const std :: string & key, const std :: string & def) const
{
  return (this->map.find(key) != this->map.end() ) ?
          this->map.at(key)                        :
          def;
}

template < typename type >
type config :: get (__unused const std :: string & key, __unused const type & def) const
{
  static_assert (std :: is_same < type, int > :: value ||
                 std :: is_same < type, float > :: value ||
                 std :: is_same < type, std :: string > :: value,
                 "Type variable not recognized! Possible variables are only [int, float, string].");

  return def; // just to avoid the warning...
}

} // end namespace
