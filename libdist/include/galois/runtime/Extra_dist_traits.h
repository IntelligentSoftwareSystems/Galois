/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

/**
 * @file Extra_dist_traits.h
 *
 * Defines particular traits used by the distributed runtime.
 */

#ifndef GALOIS_RUNTIME_EXTRA_TRAITS_H
#define GALOIS_RUNTIME_EXTRA_TRAITS_H

#include <type_traits>
#include <boost/mpl/has_xxx.hpp>

// depending on compiler version, trivially copyable defintion changes
#if __GNUC__ < 5
//! Defines what it means to be trivially copyable
#define __is_trivially_copyable(type) __has_trivial_copy(type)
#else
//! Defines what it means to be trivially copyable
#define __is_trivially_copyable(type)                                          \
        std::is_trivially_copyable < type > ::value
#endif

namespace galois {
namespace runtime {

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_has_serialize)
//! Indicates if T has the serialize trait
template <typename T>
struct has_serialize : public has_tt_has_serialize<T> {};

BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_is_copyable)
//! Indicates if T is trivially copyable
template <typename T>
struct is_copyable : public has_tt_is_copyable<T> {};

//! Indicates if T is serializable
template <typename T>
struct is_serializable {
  //! true if T is serializable
  static const bool value = has_serialize<T>::value || is_copyable<T>::value ||
                            __is_trivially_copyable(T);
};

//! Indicates if T is memory copyable
template <typename T>
struct is_memory_copyable {
  //! true if T is memory copyable
  static const bool value = is_copyable<T>::value || __is_trivially_copyable(T);
};

} // namespace runtime
} // namespace galois

#endif
