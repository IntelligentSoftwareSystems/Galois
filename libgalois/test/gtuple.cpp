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

#include "galois/gtuple.h"
#include "galois/Traits.h"
#include <iostream>
#include <cassert>

void print(std::tuple<> tpl) { std::cout << "\n"; }

template <typename T, typename... Ts>
void print(std::tuple<T, Ts...> tpl) {
  std::cout << std::get<0>(tpl) << " ";
  print(galois::tuple_cdr(tpl));
}

int main() {
  std::cout << "=== get_by_indicies ===\n";
  auto tpl = std::make_tuple(0, "asdf", 0.2);
  print(tpl);
  print(galois::get_by_indices(tpl, galois::int_seq<0, 2>{}));

  std::cout << "=== get_by_supertype ===\n";
  std::cout << galois::get_by_supertype<double>(tpl) << "\n";
  std::cout << galois::get_by_supertype<double>(std::make_tuple(1, "aaaa", 0.2))
            << "\n";
  static_assert(!galois::exists_by_supertype<double, std::tuple<int>>::value,
                "failure with missing element");
  static_assert(!galois::exists_by_supertype<double, std::tuple<>>::value,
                "failure with missing element in empty tuple");
  static_assert(galois::exists_by_supertype<int, std::tuple<int>>::value,
                "failure with existing element");
  std::cout << galois::get_by_supertype<int>(std::make_tuple(0, 0.0)) << "\n";

  std::cout << "=== get_default_trait_values ===\n";
  print(galois::get_default_trait_values(
      std::make_tuple(0, "aaaa"), std::make_tuple(0.2), std::make_tuple(0.2)));
  print(galois::get_default_trait_values(std::make_tuple(1, "aaaa", 0.2),
                                         std::make_tuple(0.2),
                                         std::make_tuple(0.2)));
  print(galois::get_default_trait_values(
      std::make_tuple(), std::make_tuple(0.2), std::make_tuple(0.2)));
  static_assert(
      galois::exists_by_supertype<
          double, decltype(galois::get_default_trait_values(
                      std::make_tuple(0, "aaaa"), std::make_tuple(0.2),
                      std::make_tuple(0.2)))>::value,
      "get_default_trait_values should have added double element");

  return 0;
}
