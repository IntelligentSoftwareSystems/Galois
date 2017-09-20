#ifndef GALOIS_ORDERED_TRAITS_H
#define GALOIS_ORDERED_TRAITS_H

#include "galois/Traits.h"
#include "galois/gtuple.h"

namespace galois {

struct enable_parameter_tag {};
template <bool V=false>
struct enable_parameter: public trait_has_svalue<bool, V>, enable_parameter_tag {};

struct has_exec_function_tag {};
template <typename T=bool>
struct has_exec_function: public trait_has_type<T>, has_exec_function_tag {};

struct needs_custom_locking_tag {};
template <typename T=bool>
struct needs_custom_locking: public trait_has_type<T>, needs_custom_locking_tag {};




// struct rt_chunk_size_tag {};
// 
// /**
 // * specify chunk size for do_all_coupled & do_all_choice at run time
 // * chunk size is regulated to be within [chunk_size_tag::MIN, chunk_size_tag::MAX]
 // */
// struct rt_chunk_size: public trait_has_value<const unsigned>, rt_chunk_size_tag {
  // unsigned regulate (const unsigned cs) const {
    // return std::min (std::max (unsigned (chunk_size_tag::MIN), cs), unsigned (chunk_size_tag::MAX));
  // }
// 
  // rt_chunk_size (const unsigned cs): trait_has_value<const unsigned> (regulate (cs)) {} 
// };


} // end namespace galois

// TODO: add 
#endif // GALOIS_ORDERED_TRAITS_H
