/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

/*
 * utilString.cpp
 *
 *  Created on: 22/07/2014
 *      Author: jodymaick
 */

#include <cstdio>
#include <stdarg.h>
#include <regex>
#include "utilString.h"

void split(const std::string& str, const std::string& delim,
           std::vector<std::string>& parts) {
  size_t start, end = 0;
  while (end < str.size()) {
    start = end;
    while (start < str.size() &&
           (delim.find(str[start]) != std::string::npos)) {
      start++; // skip initial whitespace
    }
    end = start;
    while (end < str.size() && (delim.find(str[end]) == std::string::npos)) {
      end++; // skip to end of word
    }
    if (end - start != 0) { // just ignore zero-length strings.
      parts.push_back(std::string(str, start, end - start));
    }
  }
}

/*
std::vector<std::string> regex_split(const std::string & s, std::string rgx_str)
{ std::vector<std::string> elems;

    std::regex rgx (rgx_str);

    std::sregex_token_iterator iter(s.begin(), s.end(), rgx, -1);
    std::sregex_token_iterator end;

    while (iter != end)  {
        //std::cout << "S43:" << *iter << std::endl;
        elems.push_back(*iter);
        ++iter;
    }

    return elems;
}
*/

bool startsWith(std::string str, std::string part) {
  for (unsigned int i = 0; i < part.size(); i++)
    if (str.at(i) != part.at(i))
      return false;
  return true;
}

bool endsWith(std::string str, std::string part) {
  if (str.size() < part.size())
    return false;
  for (unsigned int i = str.size() - part.size(), j = 0; j < part.size();
       i++, j++)
    if (str.at(i) != part.at(j))
      return false;
  return true;
}

std::string format(const std::string fmt, ...) {
  int size = 100;
  std::string str;
  va_list ap;
  while (1) {
    str.resize(size);
    va_start(ap, fmt);
    int n = vsnprintf((char*)str.c_str(), size, fmt.c_str(), ap);
    va_end(ap);
    if (n > -1 && n < size) {
      str.resize(n);
      return str;
    }
    if (n > -1)
      size = n + 1;
    else
      size *= 2;
  }
  return str;
}

void find_and_replace(std::string& source, std::string const& find,
                      std::string const& replace) {
  for (std::string::size_type i = 0;
       (i = source.find(find, i)) != std::string::npos;) {
    source.replace(i, find.length(), replace);
    i += replace.length();
  }
}

std::string get_clean_string(std::string string) {
  find_and_replace(string, "/", "_");
  find_and_replace(string, "\\", "_");
  find_and_replace(string, ".", "_");
  find_and_replace(string, "(", "_");
  find_and_replace(string, ")", "_");
  find_and_replace(string, "[", "_");
  find_and_replace(string, "]", "_");
  return string;
}
