#include "galois/GetEnv.h"

#include <cstdlib>
#include <stdexcept>

namespace {

bool Convert(const std::string& var_val, bool* ret) {
  // TODO(ddn): strip whitespace, case-insensitive?
  if (var_val == "True" || var_val == "1" || var_val == "true") {
    *ret = true;
    return true;
  }

  if (var_val == "False" || var_val == "0" || var_val == "false") {
    *ret = false;
    return true;
  }

  return false;
}

bool Convert(const std::string& var_val, int* ret) {
  try {
    *ret = std::stoi(var_val);
  } catch (std::invalid_argument&) {
    return false;
  } catch (std::out_of_range&) {
    return false;
  }
  return true;
}

bool Convert(const std::string& var_val, double* ret) {
  try {
    *ret = std::stod(var_val);
  } catch (std::invalid_argument&) {
    return false;
  } catch (std::out_of_range&) {
    return false;
  }
  return true;
}

bool Convert(const std::string& var_val, std::string* ret) {
  *ret = var_val;
  return true;
}

template <typename T>
bool GenericGetEnv(const std::string& var_name, T* ret) {
  char* var_val = std::getenv(var_name.c_str());
  if (!var_val) {
    return false;
  }
  return Convert(var_val, ret);
}

} // namespace

bool galois::GetEnv(const std::string& var_name, bool* ret) {
  return GenericGetEnv(var_name, ret);
}

bool galois::GetEnv(const std::string& var_name, int* ret) {
  return GenericGetEnv(var_name, ret);
}

bool galois::GetEnv(const std::string& var_name, std::string* ret) {
  return GenericGetEnv(var_name, ret);
}

bool galois::GetEnv(const std::string& var_name, double* ret) {
  return GenericGetEnv(var_name, ret);
}

bool galois::GetEnv(const std::string& var_name) {
  return std::getenv(var_name.c_str()) != nullptr;
}
