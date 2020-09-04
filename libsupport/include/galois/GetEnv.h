#ifndef GALOIS_LIBSUPPORT_GALOIS_GET_ENV_H_
#define GALOIS_LIBSUPPORT_GALOIS_GET_ENV_H_

#include <string>

namespace galois {

/// Return true if the environment variable is set.
///
/// This function simply tests for the presence of an environment variable; in
/// contrast, bool GetEnv(std::string, bool&) checks if the value of the
/// environment variable matches common truthy and falsey values.
bool GetEnv(const std::string& var_name);

/// Return true if environment variable is set, and extract its value into
/// ret_val parameter.
///
/// \param var_name name of the variable
/// \param[out] ret_val where to store the value of environment variable
/// \return true if environment variable set and value was successfully parsed;
///   false otherwise
bool GetEnv(const std::string& var_name, bool* ret);
bool GetEnv(const std::string& var_name, int* ret);
bool GetEnv(const std::string& var_name, double* ret);
bool GetEnv(const std::string& var_name, std::string* ret);

} // end namespace galois

#endif
