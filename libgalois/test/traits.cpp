#include "galois/gIO.h"
#include "galois/Traits.h"
#include <iostream>
#include <utility>

struct A {};

struct B : public A {
  std::string name_;
  B(std::string name) : name_(std::move(name)) {}
  B() : B("") {}
};

struct Unrelated {};

template <size_t... Ints, typename Tuple>
void print(std::index_sequence<Ints...>, Tuple tup) {
  (..., (std::cout << typeid(std::get<Ints>(tup)).name() << " ")) << "\n";
}

template <typename Tuple>
void print(Tuple tup) {
  print(std::make_index_sequence<std::tuple_size<Tuple>::value>(), tup);
}

int main() {
  auto pull_from_default = galois::get_default_trait_values(
      std::make_tuple(Unrelated{}), std::make_tuple(A{}), std::make_tuple(B{}));
  static_assert(
      std::is_same<decltype(pull_from_default), std::tuple<B>>::value);

  auto no_pull_from_default_when_same = galois::get_default_trait_values(
      std::make_tuple(A{}), std::make_tuple(A{}), std::make_tuple(B{}));
  static_assert(std::is_same<decltype(no_pull_from_default_when_same),
                             std::tuple<>>::value);

  auto no_pull_from_default_when_derived = galois::get_default_trait_values(
      std::make_tuple(B{}), std::make_tuple(A{}), std::make_tuple(B{}));
  static_assert(std::is_same<decltype(no_pull_from_default_when_derived),
                             std::tuple<>>::value);

  auto empty_tuple = galois::get_default_trait_values(
      std::make_tuple(), std::make_tuple(), std::make_tuple());
  static_assert(std::is_same<decltype(empty_tuple), std::tuple<>>::value);

  auto value_from_default = galois::get_default_trait_values(
      std::make_tuple(), std::make_tuple(A{}), std::make_tuple(B{"name"}));
  GALOIS_ASSERT(std::get<0>(value_from_default).name_ == "name");

  auto get_value = galois::get_trait_value<A>(std::tuple<B>(B{"name"}));
  GALOIS_ASSERT(get_value.name_ == "name");
}
