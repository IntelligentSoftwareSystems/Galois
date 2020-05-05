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

#ifndef GALOIS_RUNTIME_OPERATOR_REFERENCE_TYPES_H
#define GALOIS_RUNTIME_OPERATOR_REFERENCE_TYPES_H

#include "galois/config.h"

namespace galois {
namespace runtime {

namespace internal {

// Helper template for getting the appropriate type of
// reference to hold within each executor based off of the
// type of reference that was passed to it.

// Don't accept operators by value.
template <typename FuncTy>
struct OperatorReferenceType_impl;

// Const references are propagated.
// If a user supplies a const reference the operator() on the
// given object must be callable with *this passed as const as well.
template <typename FuncNoRef>
struct OperatorReferenceType_impl<FuncNoRef const&> {
  using type = FuncNoRef const&;
};

// Non-const references continue to be non-const.
template <typename FuncNoRef>
struct OperatorReferenceType_impl<FuncNoRef&> {
  using type = FuncNoRef&;
};

// Inside each executor store a reference to a received rvalue reference
// and then use that to pass to the various threads. This must be done in
// a way that keeps the rvalue reference alive throughout the duration of
// the parallel loop (as long as the resulting lvalue reference is used
// anywhere).
template <typename FuncNoRef>
struct OperatorReferenceType_impl<FuncNoRef&&> {
  using type = FuncNoRef&;
};

} // namespace internal

template <typename T>
using OperatorReferenceType =
    typename internal::OperatorReferenceType_impl<T>::type;

} // namespace runtime
} // namespace galois

#endif // ifndef(GALOIS_RUNTIME_OPERATOR_REFERENCE_TYPES_H)
