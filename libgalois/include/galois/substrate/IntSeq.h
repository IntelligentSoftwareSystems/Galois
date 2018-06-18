/**
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

namespace galois {
namespace substrate {

template <class T, T... I>
struct integer_sequence {
  typedef T value_type;

  static constexpr std::size_t size() noexcept;
};

namespace internal {
template <class T, T N, T Z, T... S>
struct gens : gens<T, N - 1, Z, N - 1, S...> {};
template <class T, T Z, T... S>
struct gens<T, Z, Z, S...> {
  typedef integer_sequence<T, S...> type;
};
} // namespace internal

template <std::size_t... I>
using index_sequence = integer_sequence<std::size_t, I...>;

template <class T, T N>
using make_integer_sequence =
    typename internal::gens<T, N, std::integral_constant<T, 0>::value>::type;
template <std::size_t N>
using make_index_sequence = make_integer_sequence<std::size_t, N>;

template <class... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;

} // namespace substrate
} // namespace galois
