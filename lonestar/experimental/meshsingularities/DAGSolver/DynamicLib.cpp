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
