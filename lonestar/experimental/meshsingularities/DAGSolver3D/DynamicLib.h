/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef DYNAMIC_LIB_H
#define DYNAMIC_LIB_H

#include <string>

class DLOpenError : public std::exception {
private:
  char* reason;

public:
  DLOpenError(char* what) : reason(what){};
  virtual const char* what() const throw() { return reason; }
  virtual ~DLOpenError() throw() { delete reason; }
};

class DynamicLib {

private:
  std::string path;
  void* handle = NULL;

public:
  // you can either pass handle or path for library.
  // Passing path leads to loading library and fullfill handle
  DynamicLib(const std::string& path);
  DynamicLib(const void* handle);

  ~DynamicLib();

  // load must be called before resolving symbols
  void load();
  void* resolvSymbol(const std::string& symbol);
};

#endif
