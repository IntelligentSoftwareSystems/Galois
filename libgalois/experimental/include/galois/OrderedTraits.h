/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
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
