#ifndef GALOIS_C__11_COMPAT_TYPE_TRAITS_H
#define GALOIS_C__11_COMPAT_TYPE_TRAITS_H

#include <boost/tr1/functional.hpp>
#include <boost/tr1/type_traits.hpp>

namespace std {
using namespace std::tr1;

template<bool, typename _Tp = void>
struct enable_if { };

template<typename _Tp>
struct enable_if<true, _Tp> { typedef _Tp type; };
}
#endif
