/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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
