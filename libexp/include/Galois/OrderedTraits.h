#ifndef GALOIS_ORDERED_TRAITS_H
#define GALOIS_ORDERED_TRAITS_H

#include "Galois/Traits.h"
#include "Galois/gtuple.h"

namespace Galois {

struct enable_parameter_tag {};
template <bool V=false>
struct enable_parameter: public trait_has_svalue<bool, V>, enable_parameter_tag {};

struct has_exec_function_tag {};
template <typename T=bool>
struct has_exec_function: public trait_has_type<T>, has_exec_function_tag {};

struct operator_can_abort_tag {};
template <typename T=bool>
struct operator_can_abort: public trait_has_type<T>, operator_can_abort_tag {};



struct chunk_size_tag {
  static const unsigned MIN = 1;
  static const unsigned MAX = 4096;
};

namespace HIDDEN {
  template <unsigned V, unsigned MIN, unsigned MAX> 
  struct bring_within_limits {
  private:
    static const unsigned LB = (V < MIN) ? MIN : V;
    static const unsigned UB = (LB > MAX) ? MAX : LB;

  public:
    static const unsigned value = UB;
  };

  template <unsigned V>
  struct regulate_chunk_size {
    static const unsigned value = bring_within_limits<V, chunk_size_tag::MIN, chunk_size_tag::MAX>::value;
  };
}

/**
 * specify chunk size for do_all_coupled & do_all_choice at compile time
 * chunk size is regulated to be within [chunk_size_tag::MIN, chunk_size_tag::MAX]
 */
template <unsigned SZ> 
struct chunk_size: 
  public trait_has_svalue<unsigned, HIDDEN::regulate_chunk_size<SZ>::value>, chunk_size_tag {};

struct default_chunk_size: public chunk_size<16> {};

struct rt_chunk_size_tag {};

/**
 * specify chunk size for do_all_coupled & do_all_choice at run time
 * chunk size is regulated to be within [chunk_size_tag::MIN, chunk_size_tag::MAX]
 */
struct rt_chunk_size: public trait_has_value<const unsigned>, rt_chunk_size_tag {
  unsigned regulate (const unsigned cs) const {
    return std::min (std::max (chunk_size_tag::MIN, cs), chunk_size_tag::MAX);
  }

  rt_chunk_size (const unsigned cs): trait_has_value<const unsigned> (regulate (cs)) {} 
};


} // end namespace Galois

// TODO: add 
#endif // GALOIS_ORDERED_TRAITS_H
