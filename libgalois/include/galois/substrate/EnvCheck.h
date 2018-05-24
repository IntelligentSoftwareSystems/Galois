/** Enviroment Checking Code -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_SUBSTRATE_ENVCHECK_H
#define GALOIS_SUBSTRATE_ENVCHECK_H

#include <string>

#include <cassert>

namespace galois {
namespace substrate {

namespace impl {

template <typename T>
struct ConvByType {};

template <> struct ConvByType<int> {
  static void go(const char* varVal, int& ret) {
    assert(varVal);
    ret = std::atoi(varVal);
  }
};

template <> struct ConvByType<double> {
  static void go(const char* varVal, double& ret) {
    assert(varVal);
    ret = std::atof(varVal);
  }
};

template <> struct ConvByType<std::string> {
  static void go(const char* varVal, std::string& ret) {
    assert(varVal);
    ret = varVal;
  }
};

template <typename T>
bool genericGetEnv(const char* varName, T& ret) {

  char* varVal = getenv(varName);
  if (varVal) {
    ConvByType<T>::go(varVal, ret);
    return true;
  } else {
    return false;
  }
}

} // end namespace impl

//! Return true if the Enviroment variable is set
bool EnvCheck(const char* varName);
bool EnvCheck(const std::string& varName);

/**
 * Return true if Enviroment variable is set, and extract its value into 'retVal' parameter
 * @param varName: name of the variable
 * @param retVal: lvalue to store the value of environment variable
 * @return true if environment variable set, false otherwise
 */
template <typename T>
bool EnvCheck(const char* varName, T& retVal) {
  return impl::genericGetEnv(varName, retVal);
}

template <typename T>
bool EnvCheck(const std::string& varName, T& retVal) {
  return EnvCheck(varName.c_str(), retVal);
}

} // end namespace substrate
} // end namespace galois

#endif
