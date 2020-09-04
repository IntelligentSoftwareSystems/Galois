/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_TRAITS_H
#define GALOIS_TRAITS_H

#include <tuple>
#include <type_traits>

#include "galois/config.h"
#include "galois/worklists/WorkList.h"

namespace galois {

// Trait classifications

template <typename T>
struct trait_has_type {
  typedef T type;
};

template <typename T>
struct trait_has_value {
  typedef T type;
  type value;
  trait_has_value(const type& v) : value(v) {}
  trait_has_value(type&& v) : value(std::move(v)) {}
  T getValue() const { return value; }
};

template <typename T, T V>
struct trait_has_svalue {
  typedef T type;
  static const type value = V;
  T getValue() const { return V; }
};

/**
 * Utility function to simplify creating traits that take unnamed functions
 * (i.e., lambdas).
 */
template <template <typename...> class TT, typename... Args>
auto make_trait_with_args(Args... args) -> TT<Args...> {
  return TT<Args...>(args...);
}

/**
 * True if Derived is derived from Base or is Base itself.
 *
 * A matching trait is any type that inherits from a trait.
 */
template <typename Base, typename Derived>
constexpr bool at_least_base_of =
    std::is_base_of<Base, Derived>::value || std::is_same<Base, Derived>::value;

/**
 * Returns index of first matching trait in Tuple.
 *
 * This function is not well-defined if there is no matching trait.
 */
template <typename T, typename Tuple, size_t Int, size_t... Ints>
constexpr size_t find_trait(std::index_sequence<Int, Ints...> /*seq*/) {
  if constexpr (at_least_base_of<
                    T, typename std::tuple_element<Int, Tuple>::type>) {
    return Int;
  } else {
    return find_trait<T, Tuple>(std::index_sequence<Ints...>{});
  }
}

template <typename T, typename Tuple>
constexpr size_t find_trait() {
  constexpr std::make_index_sequence<std::tuple_size<Tuple>::value> seq{};
  return find_trait<T, Tuple>(seq);
}

/**
 * Returns true if the tuple type contains the given trait T.
 */
template <typename T, typename... Ts>
constexpr bool has_trait(std::tuple<Ts...>* /*tpl*/) {
  return (... || at_least_base_of<T, Ts>);
}

template <typename T, typename Tuple>
constexpr bool has_trait() {
  return has_trait<T>(static_cast<Tuple*>(nullptr));
}

/**
 * Returns the value associated with the given trait T in a tuple.
 *
 * This function is not well-defined when there is not matching trait.
 */
template <typename T, typename Tuple>
constexpr auto get_trait_value(Tuple tpl) {
  constexpr size_t match(find_trait<T, Tuple>());
  return std::get<match>(tpl);
}

/**
 * Returns the type associated with the given trait in a tuple.
 */
template <typename T, typename Tuple>
struct get_trait_type {
  using type = typename std::tuple_element<find_trait<T, Tuple>(), Tuple>::type;
};

// Fallback to enable_if tricks over if constexpr to play more nicely with
// unused parameter warnings.

template <typename S, typename T, typename D>
constexpr auto get_default_trait_value(
    S /*source*/, T /*tag*/, D /*def*/,
    typename std::enable_if<has_trait<T, S>()>::type* = nullptr) {
  return std::make_tuple();
}

template <typename S, typename T, typename D>
constexpr auto get_default_trait_value(
    S GALOIS_UNUSED(source), T GALOIS_UNUSED(tags), D defaults,
    typename std::enable_if<!has_trait<T, S>()>::type* = nullptr) {
  return std::make_tuple(defaults);
}

/**
 * Returns a tuple that has an element from defaults[i] for every type
 * from tags[i] missing in source.
 */
template <typename S, typename T, typename D>
constexpr auto
get_default_trait_values(std::index_sequence<> GALOIS_UNUSED(seq),
                         S GALOIS_UNUSED(source), T GALOIS_UNUSED(tags),
                         D GALOIS_UNUSED(defaults)) {
  return std::make_tuple();
}

template <size_t... Ints, typename S, typename T, typename D>
constexpr auto
get_default_trait_values(std::index_sequence<Ints...> GALOIS_UNUSED(seq),
                         S source, T tags, D defaults) {
  return std::tuple_cat(get_default_trait_value(source, std::get<Ints>(tags),
                                                std::get<Ints>(defaults))...);
}

template <typename S, typename T, typename D>
constexpr auto get_default_trait_values(S source, T tags, D defaults) {
  constexpr std::make_index_sequence<std::tuple_size<T>::value> seq{};
  return get_default_trait_values(seq, source, tags, defaults);
}

template <typename T>
constexpr auto has_function_traits(int)
    -> decltype(std::declval<typename T::function_traits>(), bool()) {
  return true;
}

template <typename>
constexpr auto has_function_traits(...) -> bool {
  return false;
}

template <typename T, typename Enable = void>
struct function_traits {
  typedef std::tuple<> type;
};

template <typename T>
struct function_traits<
    T, typename std::enable_if<has_function_traits<T>(0)>::type> {
  typedef typename T::function_traits type;
};

// Traits

/**
 * Indicate name to appear in statistics. Optional argument to {@link do_all()}
 * and {@link for_each()} loops.
 */
struct loopname_tag {};
struct loopname : public trait_has_value<const char*>, loopname_tag {
  loopname(const char* p = "ANON_LOOP") : trait_has_value<const char*>(p) {}
};

/**
 * Indicate whether @{link do_all()} loops should perform work-stealing.
 * Optional argument to {@link do_all()} loops.
 */
struct steal_tag {};
struct steal : public trait_has_type<bool>, steal_tag {};

/**
 * Indicates worklist to use. Optional argument to {@link for_each()} loops.
 */
struct wl_tag {};
template <typename T, typename... Args>
struct s_wl : public trait_has_type<T>, wl_tag {
  std::tuple<Args...> args;
  s_wl(Args&&... a) : args(std::forward<Args>(a)...) {}
};

template <typename T, typename... Args>
s_wl<T, Args...> wl(Args&&... args) {
  return s_wl<T, Args...>(std::forward<Args>(args)...);
}

//
/**
 * Indicates the operator may request the parallel loop to be suspended and a
 * given function run in serial
 */
struct parallel_break_tag {};
struct parallel_break : public trait_has_type<bool>, parallel_break_tag {};

/**
 * Indicates the operator does not generate new work and push it on the worklist
 */
struct no_pushes_tag {};
struct no_pushes : public trait_has_type<bool>, no_pushes_tag {};

/**
 * Indicates the operator may request the access to a per-iteration allocator
 */
struct per_iter_alloc_tag {};
struct per_iter_alloc : public trait_has_type<bool>, per_iter_alloc_tag {};

/**
 * Indicates the operator doesn't need its execution stats recorded
 */
struct no_stats_tag {};
struct no_stats : public trait_has_type<bool>, no_stats_tag {};

/**
 * Indicates the operator needs detailed stats
 * Must provide loopname to enable this flag
 */
struct more_stats_tag {};
struct more_stats : public trait_has_type<bool>, more_stats_tag {};

/**
 * Indicates the operator doesn't need abort support
 */
struct disable_conflict_detection_tag {};
struct disable_conflict_detection : public trait_has_type<bool>,
                                    disable_conflict_detection_tag {};

/**
 * Indicates that the neighborhood set does not change through out i.e. is not
 * dependent on computed values. Examples of such fixed neighborhood is e.g.
 * the neighborhood being all the neighbors of a node in the input graph,
 * while the counter example is the neighborhood being some of the neighbors
 * based on some predicate.
 */
struct fixed_neighborhood_tag {};
struct fixed_neighborhood : public trait_has_type<bool>,
                            fixed_neighborhood_tag {};

/**
 * Indicates that the operator uses the intent to read flag.
 */
struct intent_to_read_tag {};
struct intent_to_read : public trait_has_type<bool>, intent_to_read_tag {};

/**
 * Indicates the operator has a function that visits the neighborhood of the
 * operator without modifying it.
 */
struct neighborhood_visitor_tag {};
template <typename T>
struct neighborhood_visitor : public trait_has_value<T>,
                              neighborhood_visitor_tag {
  neighborhood_visitor(const T& t = T{}) : trait_has_value<T>(t) {}
  neighborhood_visitor(T&& t) : trait_has_value<T>(std::move(t)) {}
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
template <typename T>
struct det_parallel_break : public trait_has_value<T>, det_parallel_break_tag {
  static_assert(std::is_same<typename std::result_of<T()>::type, bool>::value,
                "signature must be bool()");
  det_parallel_break(const T& t = T()) : trait_has_value<T>(t) {}
  det_parallel_break(T&& t) : trait_has_value<T>(std::move(t)) {}
};

/**
 * Indicates the operator has a function that optimizes the generation of
 * unique ids for active elements. This function should be thread-safe.
 *
 * The function should have the signature <code>uintptr_t (A)</code> where
 * A is the type of active elements.
 */
struct det_id_tag {};
template <typename T>
struct det_id : public trait_has_value<T>, det_id_tag {
  det_id(const T& t = T()) : trait_has_value<T>(t) {}
  det_id(T&& t) : trait_has_value<T>(std::move(t)) {}
};

/**
 * Indicates the operator has a type that encapsulates state that is passed
 * between the suspension and resumpsion of an operator during deterministic
 * scheduling.
 */
struct local_state_tag {};
template <typename T>
struct local_state : public trait_has_type<T>, local_state_tag {};

// TODO: separate to libdist
/** For distributed Galois **/
struct op_tag {};

struct chunk_size_tag {
  enum { MIN = 1, MAX = 4096 };
};

/**
 * Specify chunk size for do_all_coupled & do_all_choice at compile time or at
 * runtime.
 *
 * For compile time, use the template argument, e.g., galois::chunk_size<16> ()
 * Additionally, user may provide a runtime argument, e.g,
 * galois::chunk_size<16> (8)
 *
 * Currently, only do_all_coupled can take advantage of the runtime argument.
 * TODO: allow runtime provision/tuning of chunk_size in other loop executors
 *
 * chunk size is clamped to within [chunk_size_tag::MIN, chunk_size_tag::MAX]
 */
template <unsigned SZ = 32>
struct chunk_size : public trait_has_value<unsigned>, chunk_size_tag {
private:
  constexpr static unsigned clamp(unsigned int v) {
    return std::min(std::max(v, unsigned{chunk_size_tag::MIN}),
                    unsigned{chunk_size_tag::MAX});
  }

public:
  constexpr static unsigned value = clamp(SZ);

  chunk_size(unsigned cs = SZ) : trait_has_value(clamp(cs)) {}
};

typedef worklists::PerSocketChunkFIFO<chunk_size<>::value> defaultWL;

namespace internal {

template <typename Tup>
struct NeedStats {
  constexpr static const bool value =
      !has_trait<no_stats_tag, Tup>() && has_trait<loopname_tag, Tup>();
};

template <typename Tup>
std::enable_if_t<has_trait<loopname_tag, Tup>(), const char*>
getLoopName(const Tup& t) {
  return get_trait_value<loopname_tag>(t).value;
}

template <typename Tup>
std::enable_if_t<!has_trait<loopname_tag, Tup>(), const char*>
getLoopName(const Tup&) {
  return "ANON_LOOP";
}
} // namespace internal

} // namespace galois

#endif
