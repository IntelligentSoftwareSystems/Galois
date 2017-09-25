/** Galois traits -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2014, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * Optional properties (type traits) for {@link galois::for_each()}, {@link
 * galois::do_all()}, etc. can be supplied in two ways.
 *
 * First, by passing an argument to the corresponding method call:
 * \code
 * galois::for_each(v.begin(), v.end(), fn, galois::parallel_break());
 * \endcode
 *
 * Second, by providing a specially named nested type
 * \code
 * #include <tuple>
 *
 * struct MyClass { 
 *   typedef std::tuple<galois::parallel_break> function_traits;
 * };
 *
 * int main() {
 *   galois::for_each(v.begin(), v.end(), MyClass {});
 * \endcode
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_TRAITS_H
#define GALOIS_TRAITS_H

#include "galois/gtuple.h"

#include <type_traits>
#include <tuple>

namespace galois {

//! @section Trait classifications

template<typename T>
struct trait_has_type {
  typedef T type;
};

template<typename T>
struct trait_has_value {
  typedef T type;
  type value;
  trait_has_value(const type& v): value(v) {}
  trait_has_value(type&& v): value(std::move(v)) {}
  T getValue() const { return value; }
};

template<typename T, T V>
struct trait_has_svalue {
  typedef T type;
  static const type value = V;
  T getValue() const { return V; }
};

//! @section Utility

/**
 * Utility function to simplify creating traits that take unnamed functions
 * (i.e., lambdas).
 */
template<template<typename...> class TT, typename... Args>
auto make_trait_with_args(Args... args) -> TT<Args...> {
  return TT<Args...>(args...);
}

namespace internal {

template<typename Tuple, typename TagsTuple, int... Is>
struct indices_of_non_matching_tags_aux {
  typedef int_seq<> type; 
};

template<bool Match, typename Tuple, typename TagsTuple, int I, int... Is>
struct apply_indices_of_non_matching_tags {
  typedef typename indices_of_non_matching_tags_aux<Tuple, TagsTuple, Is...>::type type; 
};

template<typename Tuple, typename TagsTuple, int I, int... Is>
struct apply_indices_of_non_matching_tags<false, Tuple, TagsTuple, I, Is...> {
  typedef typename indices_of_non_matching_tags_aux<Tuple, TagsTuple, Is...>::type::template append<I>::type type;
};

template<typename Tuple, typename TagsTuple, int I, int... Is>
struct indices_of_non_matching_tags_aux<Tuple, TagsTuple, I, Is...> {
  static const bool matches = exists_by_supertype<typename std::tuple_element<I, TagsTuple>::type, Tuple>::value;
  typedef typename apply_indices_of_non_matching_tags<matches, Tuple, TagsTuple, I, Is...>::type type;
};

template<typename Tuple, typename TagsTuple, typename Seq>
struct indices_of_non_matching_tags { 
  typedef int_seq<> type; 
};

template<typename Tuple, typename TagsTuple, int I, int... Is>
struct indices_of_non_matching_tags<Tuple, TagsTuple, int_seq<I, Is...> > {
  static const bool matches = exists_by_supertype<typename std::tuple_element<I, TagsTuple>::type, Tuple>::value;
  typedef typename apply_indices_of_non_matching_tags<matches, Tuple, TagsTuple, I, Is...>::type type;
};

}

/**
 * Returns a tuple that has an element from defaults[i] for every type 
 * from tags[i] missing in source.
 */
template<typename S, typename T, typename D,
  typename Seq = typename make_int_seq<std::tuple_size<T>::value>::type,
  typename ResSeq = typename internal::indices_of_non_matching_tags<S,T,Seq>::type>
typename tuple_elements<D, ResSeq>::type
get_default_trait_values(S source, T tags, D defaults)
{
  return get_by_indices(defaults, ResSeq {});
}

template<typename T, typename Tuple, 
  typename Seq = typename make_int_seq<std::tuple_size<Tuple>::value - 1>::type>
typename tuple_elements<Tuple, Seq>::type  
get_tuple_without(T rm_type, Tuple tpl)
{
  typedef typename make_int_seq<subtype_index_nodup<T, Tuple>::value>::type Seq_pre;
  typedef typename make_int_seq<std::tuple_size<Tuple>::value - subtype_index_nodup<T, Tuple>::value - 1>::type Seq_post;
  return std::tuple_cat(get_by_offset<0>(tpl, Seq_pre{}), get_by_offset<subtype_index_nodup<T, Tuple>::value + 1>(tpl, Seq_post{})); 
}

template<typename T>
constexpr auto has_function_traits(int) -> decltype(std::declval<typename T::function_traits>(), bool()) {
  return true;
}

template<typename>
constexpr auto has_function_traits(...) -> bool {
  return false;
}

template<typename T, typename Enable = void>
struct function_traits {
  typedef std::tuple<> type;
};

template<typename T>
struct function_traits<T, typename std::enable_if<has_function_traits<T>(0)>::type> {
  typedef typename T::function_traits type;
};

//! @section Traits



/**
 * Indicate name to appear in statistics. Optional argument to {@link do_all()}
 * and {@link for_each()} loops.
 */
struct loopname_tag {};
struct loopname: public trait_has_value<const char*>, loopname_tag {
  loopname(const char* p = nullptr): trait_has_value<const char*>(p) {
    if (p == nullptr) { p = "ANON_LOOP"; }
  }
};

struct default_loopname: public loopname {
  default_loopname (void): loopname ("ANON_LOOP") {}
};

/**
   - * Indicate run-number to appear in statistics. Optional argument to {@link do_all()}
   - */
// TODO remove numrun
struct numrun_tag {};
struct numrun: public trait_has_value<const std::string>, numrun_tag {
   numrun(const std::string& p = ""): trait_has_value<const std::string>(p) { }
};


/**
 * Indicate whether @{link do_all()} loops should perform work-stealing. Optional
 * argument to {@link do_all()} loops.
 */
struct do_all_steal_tag {};
template<bool V = false>
struct do_all_steal: public trait_has_svalue<bool, V>, do_all_steal_tag {};

/**
 * Indicates worklist to use. Optional argument to {@link for_each()} loops.
 */
struct wl_tag {};
template<typename T, typename... Args>
struct s_wl: public trait_has_type<T>, wl_tag {
  std::tuple<Args...> args;
  s_wl(Args&&... a): args(std::forward<Args>(a)...) {}
};

template<typename T, typename... Args>
s_wl<T, Args...> wl(Args&&... args) {
  return s_wl<T, Args...>(std::forward<Args>(args)...);
}

/**
 * Presence of timeit option will automatically time the do_all or for_each loop 
 * */
struct timeit_tag {};
struct timeit: public trait_has_type<bool>, timeit_tag {};

//
/**
 * Indicates the operator may request the parallel loop to be suspended and a
 * given function run in serial
 */
struct parallel_break_tag {};
struct parallel_break: public trait_has_type<bool>, parallel_break_tag {};

/**
 * Indicates the operator does not generate new work and push it on the worklist
 */
struct no_pushes_tag {};
struct no_pushes: public trait_has_type<bool>, no_pushes_tag {};

/**
 * Indicates the operator may request the access to a per-iteration allocator
 */
struct per_iter_alloc_tag {};
struct per_iter_alloc: public trait_has_type<bool>, per_iter_alloc_tag {};

/**
 * Indicates the operator doesn't need its execution stats recorded
 */
struct no_stats_tag {};
struct no_stats: public trait_has_type<bool>, no_stats_tag { };

/**
 * Indicates the operator needs detailed stats
 */
struct more_stats_tag {};
struct more_stats: public trait_has_type<bool>, more_stats_tag { };

/**
 * Indicates the operator doesn't need abort support
 */
struct no_conflicts_tag {};
struct no_conflicts: public trait_has_type<bool>, no_conflicts_tag {};

/**
 * Indicates that the neighborhood set does not change through out i.e. is not
 * dependent on computed values. Examples of such fixed neighborhood is e.g.
 * the neighborhood being all the neighbors of a node in the input graph,
 * while the counter example is the neighborhood being some of the neighbors
 * based on some predicate. 
 */
struct fixed_neighborhood_tag {};
struct fixed_neighborhood: public trait_has_type<bool>, fixed_neighborhood_tag {};

/**
 * Indicates that the operator uses the intent to read flag.
 */
struct intent_to_read_tag {};
struct intent_to_read: public trait_has_type<bool>, intent_to_read_tag {};

/**
 * Indicates the operator has a function that visits the neighborhood of the
 * operator without modifying it.
 */
struct neighborhood_visitor_tag {};
template<typename T>
struct neighborhood_visitor: public trait_has_value<T>, neighborhood_visitor_tag {
  neighborhood_visitor(const T& t = T {}): trait_has_value<T>(t) {}
  neighborhood_visitor(T&& t): trait_has_value<T>(std::move(t)) {}
};

/**
 * Indicates the operator has a function that allows a {@link
 * galois::for_each} loop to be exited deterministically.
 *
 * The function should have the signature <code>bool()</code>. 
 *
 * It will be periodically called by the deterministic scheduler.  If it
 * returns true, the loop ends as if calling {@link UserContext::breakLoop},
 * but unlike that function, these breaks are deterministic.
 */
struct det_parallel_break_tag {};
template<typename T>
struct det_parallel_break: public trait_has_value<T>, det_parallel_break_tag {
  static_assert(std::is_same<typename std::result_of<T()>::type, bool>::value, "signature must be bool()");
  det_parallel_break(const T& t = T {}): trait_has_value<T>(t) {}
  det_parallel_break(T&& t): trait_has_value<T>(std::move(t)) {}
};

/**
 * Indicates the operator has a function that optimizes the generation of
 * unique ids for active elements. This function should be thread-safe.
 *
 * The function should have the signature <code>uintptr_t (A)</code> where
 * A is the type of active elements.
 *
 * An example of use:
 *
 * \snippet test/deterministic.cpp Id
 */
struct det_id_tag {};
template<typename T>
struct det_id: public trait_has_value<T>, det_id_tag {
  det_id(const T& t = T {}): trait_has_value<T>(t) {}
  det_id(T&& t): trait_has_value<T>(std::move(t)) {}
};

/**
 * Indicates the operator has a type that encapsulates state that is passed between 
 * the suspension and resumpsion of an operator during deterministic scheduling.
 *
 * An example of use:
 *
 * \snippet test/deterministic.cpp Local state
 */
struct local_state_tag {};
template<typename T>
struct local_state: public trait_has_type<T>, local_state_tag {};


// TODO: separate to libdist
/** For distributed Galois **/
struct op_tag {};

/**
 * Array of uint32_t. Could be used for various things, but main use case now
 * is to pass in information for thread work splitting.
 */
// TODO: delete this feature
struct thread_range_tag {};

struct thread_range: public trait_has_value<const uint32_t*>, thread_range_tag {
  thread_range(const uint32_t* t = nullptr) : 
    trait_has_value<const uint32_t*>(t) {}
};

struct chunk_size_tag {
  enum { MIN = 1, MAX = 4096 };
};


namespace internal {
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
 * For compile time, use the template argument, e.g., galois::chunk_size<16> ()
 * Additionally, user may provide a runtime argument, e.g, galois::chunk_size<16> (8)
 *
 * Currently, only do_all_coupled can take advantage of the runtime argument. 
 * TODO: allow runtime provision/tuning of chunk_size in other loop executors
 *
 * chunk size is regulated to be within [chunk_size_tag::MIN, chunk_size_tag::MAX]
 */

template <unsigned SZ> 
struct chunk_size: 
  // public trait_has_svalue<unsigned, internal::regulate_chunk_size<SZ>::value>, trait_has_value<unsigned>, chunk_size_tag {    
  public trait_has_value<unsigned>, chunk_size_tag {    

  static const unsigned value = internal::regulate_chunk_size<SZ>::value;

  unsigned regulate (const unsigned cs) const {
    return std::min (std::max (unsigned (chunk_size_tag::MIN), cs), unsigned (chunk_size_tag::MAX));
  }

  chunk_size (unsigned cs=SZ): trait_has_value (regulate (cs)) {}
};

struct default_chunk_size: public chunk_size<16> {};
} // close namespace galois


#endif
