#ifndef GALOIS_ORDERED_TRAITS_H
#define GALOIS_ORDERED_TRAITS_H

#include "galois/Traits.h"
#include "galois/gtuple.h"

namespace galois {

struct enable_parameter_tag {};
template <bool V = false>
struct enable_parameter : public trait_has_svalue<bool, V>,
                          enable_parameter_tag {};

struct has_exec_function_tag {};
template <typename T = bool>
struct has_exec_function : public trait_has_type<T>, has_exec_function_tag {};

struct needs_custom_locking_tag {};
template <typename T = bool>
struct needs_custom_locking : public trait_has_type<T>,
                              needs_custom_locking_tag {};

} // end namespace galois

#endif // GALOIS_ORDERED_TRAITS_H
