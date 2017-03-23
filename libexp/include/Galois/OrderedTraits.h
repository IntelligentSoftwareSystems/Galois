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

struct needs_custom_locking_tag {};
template <typename T=bool>
struct needs_custom_locking: public trait_has_type<T>, needs_custom_locking_tag {};



struct chunk_size_tag {
  enum { MIN = 1, MAX = 4096 };
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
 * specify chunk size for do_all_coupled & do_all_choice at compile time or
 * at runtime
 * For compile time, use the template argument, e.g., Galois::chunk_size<16> ()
 * Additionally, user may provide a runtime argument, e.g, Galois::chunk_size<16> (8)
 *
 * Currently, only do_all_coupled can take advantage of the runtime argument. 
 * TODO: allow runtime provision/tuning of chunk_size in other loop executors
 *
 * chunk size is regulated to be within [chunk_size_tag::MIN, chunk_size_tag::MAX]
 */

template <unsigned SZ> 
struct chunk_size: 
  // public trait_has_svalue<unsigned, HIDDEN::regulate_chunk_size<SZ>::value>, trait_has_value<unsigned>, chunk_size_tag {    
  public trait_has_value<unsigned>, chunk_size_tag {    

  static const unsigned value = HIDDEN::regulate_chunk_size<SZ>::value;

  unsigned regulate (const unsigned cs) const {
    return std::min (std::max (unsigned (chunk_size_tag::MIN), cs), unsigned (chunk_size_tag::MAX));
  }

  chunk_size (unsigned cs=SZ): trait_has_value (regulate (cs)) {}
};

struct default_chunk_size: public chunk_size<16> {};

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


} // end namespace Galois

// TODO: add 
#endif // GALOIS_ORDERED_TRAITS_H
