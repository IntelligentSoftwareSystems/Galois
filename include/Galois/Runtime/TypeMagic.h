/** Type Magic -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * Implement type meta-programming stuff
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_TYPEMAGIC_H
#define GALOIS_RUNTIME_TYPEMAGIC_H

#include <tuple>

namespace Galois {
namespace Runtime {

// Cons a type onto a tuple
template <typename, typename> struct Cons;

template <typename  T, typename ...Args>
struct Cons<T, std::tuple<Args...>>
{
    using type = std::tuple<T, Args...>;
};

//Filter types into a tuple
template <template <typename> class, typename...> struct typeFilter;

template<template< typename> class Pred> struct typeFilter<Pred> { using type = std::tuple<>; };

template < template <typename> class Pred, typename Head, typename... Tail>
struct typeFilter<Pred, Head, Tail...>
{
  using type = typename std::conditional<Pred<Head>::value,
					 typename Cons<Head, typename typeFilter<Pred,Tail...>::type>::type,
					 typename typeFilter<Pred,Tail...>::type
					 >::type;
};

//apply a map to types
template < template <typename> class fMap, typename... Types>
struct typeMap
{
  using type = std::tuple<typename fMap<Types>::type...>;
};

//apply a map to types in a tuple
template < template <typename> class, typename...> struct typeTupleMap;
template < template <typename> class fMap, typename... Types>
struct typeTupleMap<fMap, std::tuple<Types...>>
{
  using type = typename typeMap<fMap, Types...>::type;
};


} // namespace Runtime
} // namespace Galois

#endif
