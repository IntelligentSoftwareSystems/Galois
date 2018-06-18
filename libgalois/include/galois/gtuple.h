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

#ifndef GALOIS_GTUPLE_H
#define GALOIS_GTUPLE_H

#include <cstddef>
#include <tuple>
#include <cstring>

namespace galois {

/** Is Derived a subtype of Base? */
template <typename Base, typename Derived>
struct is_subtype_of {
  static const bool value = std::is_base_of<Base, Derived>::value ||
                            std::is_same<Base, Derived>::value;
};

/**
 * Returns index of last tuple element that has type T or is derived from T.
 * Value is std::tuple_size<Tuple> when there is no such element.
 */
template <typename T, typename Tuple,
          size_t Index = std::tuple_size<Tuple>::value>
struct subtype_index {
private:
  typedef typename std::tuple_element<Index - 1, Tuple>::type current;

public:
  static const size_t value = is_subtype_of<T, current>::value
                                  ? Index - 1
                                  : subtype_index<T, Tuple, Index - 1>::value;
};

template <typename T, typename Tuple>
struct subtype_index<T, Tuple, 0> {
  static const size_t value = std::tuple_size<Tuple>::value;
};

/**
 * Like {@link subtype_index} but fail to compile if there are multiple matching
 * elements.
 */
template <typename T, typename Tuple,
          size_t Index1   = subtype_index<T, Tuple>::value,
          size_t Index2   = subtype_index<T, Tuple, Index1>::value,
          size_t NotFound = std::tuple_size<Tuple>::value>
struct subtype_index_nodup {};

template <typename T, typename Tuple, size_t Index1, size_t NotFound>
struct subtype_index_nodup<T, Tuple, Index1, NotFound, NotFound> {
  static const size_t value = Index1;
};

// Integer Sequences
template <typename T, T... Is>
struct integer_seq {
  using type = integer_seq;
  template <T I>
  using append = integer_seq<T, Is..., I>;
};
template <bool... B>
using bool_seq = integer_seq<bool, B...>;
template <int... I>
using int_seq = integer_seq<int, I...>;
template <unsigned... I>
using uint_seq = integer_seq<unsigned, I...>;
template <size_t... I>
using index_seq = integer_seq<size_t, I...>;

namespace internal {
template <class S1, class S2>
struct log_construct;

template <typename T, T... I1, T... I2>
struct log_construct<integer_seq<T, I1...>, integer_seq<T, I2...>>
    : integer_seq<T, I1..., (sizeof...(I1) + I2)...> {};
} // namespace internal

template <typename T, T N>
struct make_integer_seq
    : internal::log_construct<
          typename make_integer_seq<T, N / 2>::type,
          typename make_integer_seq<T, N - N / 2>::type>::type {};

template <>
struct make_integer_seq<int, 0> : integer_seq<int> {};
template <>
struct make_integer_seq<int, 1> : integer_seq<int, 0> {};
template <>
struct make_integer_seq<unsigned, 0U> : integer_seq<unsigned> {};
template <>
struct make_integer_seq<unsigned, 1U> : integer_seq<unsigned, 0U> {};
template <>
struct make_integer_seq<size_t, 0UL> : integer_seq<size_t> {};
template <>
struct make_integer_seq<size_t, 1UL> : integer_seq<size_t, 0UL> {};

template <int N>
using make_int_seq = make_integer_seq<int, N>;
template <unsigned N>
using make_uint_seq = make_integer_seq<unsigned, N>;
template <size_t N>
using make_index_seq = make_integer_seq<size_t, N>;

/**
 * std::tuple_elements generalized over int_seq.
 */
template <typename Tuple, typename... Args>
struct tuple_elements;

template <typename Tuple, int... Is>
struct tuple_elements<Tuple, int_seq<Is...>> {
  typedef std::tuple<typename std::tuple_element<Is, Tuple>::type...> type;
};

// Accessors

/**
 * Like \code std::get<T>(std::tuple<Types...> t) \endcode (C++14) except that
 * it returns an element that is derived from T. And like std::get, this fails
 * to compile when T is not in Types.
 */
template <typename T, typename... Ts>
auto get_by_supertype(std::tuple<Ts...>& tpl) -> typename std::tuple_element<
    subtype_index_nodup<T, std::tuple<Ts...>>::value,
    std::tuple<Ts...>>::type& {
  return std::get<subtype_index_nodup<T, std::tuple<Ts...>>::value>(tpl);
}

template <typename T, typename... Ts>
auto get_by_supertype(const std::tuple<Ts...>& tpl) ->
    typename std::tuple_element<
        subtype_index_nodup<T, std::tuple<Ts...>>::value,
        std::tuple<Ts...>>::type const& {
  return std::get<subtype_index_nodup<T, std::tuple<Ts...>>::value>(tpl);
}

template <typename T, typename... Ts>
auto get_by_supertype(std::tuple<Ts...>&& tpl) -> typename std::tuple_element<
    subtype_index_nodup<T, std::tuple<Ts...>>::value,
    std::tuple<Ts...>>::type&& {
  return std::move(
      std::get<subtype_index_nodup<T, std::tuple<Ts...>>::value>(tpl));
}

template <typename... Ts, int... Is>
auto get_by_indices(const std::tuple<Ts...>& tpl, int_seq<Is...>) ->
    typename tuple_elements<std::tuple<Ts...>, int_seq<Is...>>::type {
  return std::make_tuple(std::get<Is>(tpl)...);
}

template <typename... Ts, int... Is>
auto get_by_indices(std::tuple<Ts...>&& tpl, int_seq<Is...>) ->
    typename tuple_elements<std::tuple<Ts...>, int_seq<Is...>>::type&& {
  return std::move(std::make_tuple(std::get<Is>(tpl)...));
}

template <int Offset, typename... Ts, int... Is>
auto get_by_offset(const std::tuple<Ts...>& tpl, int_seq<Is...>) ->
    typename tuple_elements<std::tuple<Ts...>, int_seq<Is...>>::type {
  return std::make_tuple(std::get<Is + Offset>(tpl)...);
}

template <int Offset, typename... Ts, int... Is>
auto get_by_offset(std::tuple<Ts...>&& tpl, int_seq<Is...>) ->
    typename tuple_elements<std::tuple<Ts...>, int_seq<Is...>>::type&& {
  return std::move(std::make_tuple(std::get<Is + Offset>(tpl)...));
}

//! Get declared type of type T in tuple (which may be a subtype of T)
template <typename T, typename Tuple>
struct get_type_by_supertype {
  typedef typename std::tuple_element<subtype_index_nodup<T, Tuple>::value,
                                      Tuple>::type type;
};

template <typename T, typename Tuple>
struct exists_by_supertype {
  static const bool value =
      (subtype_index_nodup<T, Tuple>::value != std::tuple_size<Tuple>::value);
};

namespace internal {

template <typename T, typename... Args, int I0, int... Is>
auto tuple_cdr_impl(const std::tuple<T, Args...>& tuple, int_seq<I0, Is...>)
    -> std::tuple<Args...> {
  return std::make_tuple(std::get<Is>(tuple)...);
}

template <typename T, typename... Args, int I0, int... Is>
auto tuple_cdr_impl(std::tuple<T, Args...>&& tuple, int_seq<I0, Is...>)
    -> std::tuple<Args...>&& {
  return std::move(std::make_tuple(std::get<Is>(tuple)...));
}

} // namespace internal

template <typename T, typename... Args>
auto tuple_cdr(const std::tuple<T, Args...>& tpl) -> std::tuple<Args...> {
  return internal::tuple_cdr_impl(
      tpl, typename make_int_seq<sizeof...(Args) + 1>::type{});
}

template <typename T, typename... Args>
auto tuple_cdr(std::tuple<T, Args...>&& tpl) -> std::tuple<Args...>&& {
  return std::move(internal::tuple_cdr_impl(
      tpl, typename make_int_seq<sizeof...(Args) + 1>::type{}));
}

namespace internal {

template <typename Seq, bool...>
struct true_indices_impl {
  typedef int_seq<> type;
};

template <int I, int... Is, bool B, bool... Bs>
struct true_indices_impl<int_seq<I, Is...>, B, Bs...> {
  typedef typename std::conditional<
      B,
      typename true_indices_impl<int_seq<Is...>,
                                 Bs...>::type::template append<I>::type,
      typename true_indices_impl<int_seq<Is...>, Bs...>::type>::type type;
};

} // namespace internal

/**
 * Return indices of true elements.
 */
template <bool... Bs>
struct true_indices : public internal::true_indices_impl<
                          typename make_int_seq<sizeof...(Bs)>::type, Bs...> {};

} // namespace galois
#endif
