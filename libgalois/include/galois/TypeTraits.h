/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef GALOIS_TYPETRAITS_H
#define GALOIS_TYPETRAITS_H

#include "galois/substrate/CompilerSpecific.h"
#include <boost/mpl/has_xxx.hpp>

namespace galois {
namespace DEPRECATED {
#define GALOIS_HAS_MEM_FUNC(func, name) \
  template<typename T, typename Sig> \
  struct has_##name { \
    typedef char yes[1]; \
    typedef char no[2]; \
    template<typename U, U> struct type_check; \
    template<typename W> static yes& test(type_check<Sig, &W::func>*); \
    template<typename  > static no&  test(...); \
    static const bool value = sizeof(test<T>(0)) == sizeof(yes); \
  }

#define GALOIS_HAS_MEM_FUNC_ANY(func, name) \
  template<typename T> \
  struct has_##name { \
    typedef char yes[1]; \
    typedef char no[2]; \
    template<typename U, U> struct type_check; \
    template<typename V> struct Sig { typedef V type; };\
    template<typename W> static yes& test(type_check<typename Sig<decltype(&W::func)>::type, &W::func>*); \
    template<typename  > static no&  test(...); \
    static const bool value = sizeof(test<T>(0)) == sizeof(yes); \
  }

#define GALOIS_HAS_MEM_TYPE(func, name) \
  template<typename T> \
  struct has_##name { \
    typedef char yes[1]; \
    typedef char no[2]; \
    template<typename W> static yes& test(typename W::func*); \
    template<typename  > static no&  test(...); \
    static const bool value = sizeof(test<T>(0)) == sizeof(yes); \
  }

GALOIS_HAS_MEM_FUNC(galoisDeterministicParallelBreak, tf_deterministic_parallel_break);
/**
 * Indicates the operator has a member function that allows a {@link galois::for_each}
 * loop to be exited deterministically.
 *
 * The function has the following signature:
 * \code
 *  struct T {
 *    bool galoisDeterministicParallelBreak() {
 *      // returns true if loop should end
 *    }
 *  };
 *  \endcode
 *
 * This function will be periodically called by the deterministic scheduler.
 * If it returns true, the loop ends as if calling {@link
 * UserContext::breakLoop}, but unlike that function, these breaks are
 * deterministic.
 */
template<typename T>
struct has_deterministic_parallel_break : public has_tf_deterministic_parallel_break<T, bool(T::*)()> {};

GALOIS_HAS_MEM_FUNC_ANY(galoisDeterministicId, tf_deterministic_id);
/**
 * Indicates the operator has a member function that optimizes the generation
 * of unique ids for active elements. This function should be thread-safe.
 *
 * The type conforms to the following:
 * \code
 *  struct T {
 *    uintptr_t galoisDeterministicId(const A& item) const { 
 *      // returns a unique identifier for item
 *    }
 *  };
 * \endcode
 */
template<typename T>
struct has_deterministic_id : public has_tf_deterministic_id<T> {};

GALOIS_HAS_MEM_TYPE(GaloisDeterministicLocalState, tf_deterministic_local_state);
/**
 * Indicates the operator has a member type that encapsulates state that is passed between 
 * the suspension and resumpsion of an operator during deterministic scheduling.
 *
 * The type conforms to the following:
 * \code
 *  struct T {
 *    struct GaloisDeteministicLocalState {
 *      int x, y, z; // Local state
 *      GaloisDeterministicLocalState(T& self, galois::PerIterAllocTy& alloc) {
 *        // initialize local state
 *      }
 *    };
 *
 *    void operator()(const A& item, galois::UserContext<A>&) { 
 *      // An example of using local state
 *      typedef GaloisDeterministicLocalState LS;
 *      bool used;
 *      LS* p = (LS*) ctx.getLocalState(used);
 *      if (used) {
 *        // operator is being resumed; use p
 *      } else {
 *        // operator hasn't been suspended yet; execute normally
 *        // save state into p to be used when operator resumes
 *      }
 *    }
 *  };
 * \endcode 
 */
template<typename T>
struct has_deterministic_local_state : public has_tf_deterministic_local_state<T> {};

/**
 * Indicates the operator may request the parallel loop to be suspended and a
 * given function run in serial
 */
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_needs_parallel_break)
template<typename T>
struct needs_parallel_break : public has_tt_needs_parallel_break<T> {};

/**
 * Indicates the operator does not generate new work and push it on the worklist
 */
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_does_not_need_push)
template<typename T>
struct does_not_need_push : public has_tt_does_not_need_push<T> {};

/**
 * Indicates the operator may request the access to a per-iteration 
 * allocator
 */
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_needs_per_iter_alloc)
template<typename T>
struct needs_per_iter_alloc : public has_tt_needs_per_iter_alloc<T> {};

/**
 * Indicates the operator doesn't need its execution stats recorded
 */
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_does_not_need_stats)
template<typename T>
struct does_not_need_stats : public has_tt_does_not_need_stats<T> {};

/**
 * Indicates the operator doesn't need abort support
 */
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_does_not_need_aborts)
template<typename T>
struct does_not_need_aborts : public has_tt_does_not_need_aborts<T> {};

/**
 * Indicates that the neighborhood set does not change, i.e., it is not
 * dependent on computed values. Examples of fixed neighborhoods is the
 * neighborhood being all the neighbors of a node in the input graph, while the
 * counter example is the neighborhood being some of the neighbors based on
 * some predicate. 
 */
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_has_fixed_neighborhood)
template<typename T>
struct has_fixed_neighborhood: public has_tt_has_fixed_neighborhood<T> {};

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_has_intent_to_read)
template <typename T>
struct has_intent_to_read: public has_tt_has_intent_to_read<T> {};

/**
 * Temporary type trait for pre-C++11 compilers, which don't support exact
 * std::is_trivially_constructible. 
 */
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_has_known_trivial_constructor)
template<typename T>
struct has_known_trivial_constructor: public has_tt_has_known_trivial_constructor<T> { };

//! Decay that handles std::ref
template<typename T>
struct special_decay {
  using type = typename std::decay<T>::type;
};

template<typename T>
struct special_decay<std::reference_wrapper<T>> {
  using type = T&;
};

}
}
#endif
