#include "Galois/gtuple.h"
#include "Galois/Traits.h"
#include <iostream>
#include <cassert>

void print(std::tuple<> tpl) {
  std::cout << "\n";
}

template<typename T, typename... Ts>
void print(std::tuple<T, Ts...> tpl) {
  std::cout << std::get<0>(tpl) << " ";
  print(galois::tuple_cdr(tpl));
}

template<typename T, typename U = decltype(>
constexpr bool compiles(

int main() {
  std::cout << "=== get_by_indicies ===\n";
  auto tpl = std::make_tuple(0, "asdf", 0.2);
  print(tpl);
  print(galois::get_by_indices(tpl, galois::int_seq<0, 2> {}));
  print(std::move(galois::get_by_indices(tpl, galois::int_seq<0, 2> {})));

  std::cout << "=== get_by_supertype ===\n";
  std::cout << galois::get_by_supertype<double>(tpl) << "\n";
  std::cout << galois::get_by_supertype<double>(std::move(std::make_tuple(1, "aaaa", 0.2))) << "\n";
  static_assert(!galois::exists_by_supertype<double, std::tuple<int>>::value, "failure with missing element");
  static_assert(!galois::exists_by_supertype<double, std::tuple<>>::value, "failure with missing element in empty tuple");
  static_assert(galois::exists_by_supertype<int, std::tuple<int>>::value, "failure with existing element");
  std::cout << galois::get_by_supertype<int>(std::make_tuple(0, 0)) << "\n";

  std::cout << "=== get_default_trait_values ===\n";
  print(galois::get_default_trait_values(
          std::make_tuple(0, "aaaa"),
          std::make_tuple(0.2),
          std::make_tuple(0.2)));
  print(galois::get_default_trait_values(
          std::make_tuple(1, "aaaa", 0.2),
          std::make_tuple(0.2),
          std::make_tuple(0.2)));
  print(galois::get_default_trait_values(
          std::make_tuple(),
          std::make_tuple(0.2),
          std::make_tuple(0.2)));
  static_assert(
    galois::exists_by_supertype<double,
      decltype(galois::get_default_trait_values(
              std::make_tuple(0, "aaaa"),
              std::make_tuple(0.2),
              std::make_tuple(0.2)))>::value,
    "get_default_trait_values should have added double element");

  return 0;
}
