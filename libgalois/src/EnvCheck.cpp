#include "galois/substrate/EnvCheck.h"

#include <cstdlib>


bool galois::substrate::EnvCheck(const char* varName) {
  if (std::getenv(varName))
    return true;
  return false;
}

bool galois::substrate::EnvCheck(const std::string& varName) {
  return galois::substrate::EnvCheck(varName.c_str());
}

