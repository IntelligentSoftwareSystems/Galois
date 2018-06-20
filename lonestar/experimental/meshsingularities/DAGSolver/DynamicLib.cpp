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

#include "DynamicLib.h"
#include <dlfcn.h>
#include <cstdio>

DynamicLib::DynamicLib(const std::string& path) {
  this->path   = path;
  this->handle = dlopen(path.c_str(), RTLD_NOW);
  if (this->handle == NULL) {
    printf("error: %s\n", dlerror());
  }
  printf("Loaded: %s / %p\n", this->path.c_str(), this->handle);
}

DynamicLib::DynamicLib(const void* handle) {
  this->path   = "<handle passed>";
  this->handle = (void*)handle;
}

DynamicLib::~DynamicLib() { dlclose(this->handle); }

void DynamicLib::load() {
  if (this->handle == NULL) {
    this->handle = dlopen(this->path.c_str(), RTLD_NOW);
  }
  if (this->handle == NULL) {
    throw new DLOpenError(dlerror());
  }
#ifdef DEBUG
  fprintf(stderr, "Loaded: %s / %p\n", this->path.c_str(), this->handle);
#endif
}

void* DynamicLib::resolvSymbol(const std::string& symbol) {
  void* ptr = dlsym(this->handle, symbol.c_str());
  return ptr;
}
