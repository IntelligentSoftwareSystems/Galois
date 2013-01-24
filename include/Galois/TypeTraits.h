/** Galois type traits -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * There are two ways to declare a typetrait. First, with a typedef or other
 * valid name declaration:
 * \code
 * struct MyClass {
 *   typedef int tt_needs_parallel_break;
 *   ....
 * };
 * \endcode
 *
 * The second way is by specializing a function:
 * \code
 * namespace Galois {
 *   template<>
 *   struct needs_parallel_break<MyClass> : public boost::true_type {};
 * }
 * \endcode
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */
#ifndef GALOIS_TYPETRAITS_H
#define GALOIS_TYPETRAITS_H

#include <boost/mpl/has_xxx.hpp>
namespace Galois {

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
 * Indicates that the neighborhood set does not change through out i.e. is not
 * dependent on computed values. Examples of such fixed neighborhood is e.g. the 
 * neighborhood being all the neighbors of a node in the input graph, while the
 * counter example is the neighborhood being some of the neighbors based on
 * some predicate. 
 */
BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_has_fixed_neighborhood)
template <typename T>
struct has_fixed_neighborhood: public has_tt_has_fixed_neighborhood<T> {};

}
#endif
