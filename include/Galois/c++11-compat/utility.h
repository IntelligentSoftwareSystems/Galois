#ifndef GALOIS_C__11_COMPAT_UTILITY_H
#define GALOIS_C__11_COMPAT_UTILITY_H

#include <boost/tr1/type_traits.hpp>

namespace std {
template<typename _Tp>
constexpr _Tp&& forward(typename std::tr1::remove_reference<_Tp>::type& __t) {
  return static_cast<_Tp&&>(__t);
}

template<typename _Tp>
constexpr _Tp&& forward(typename std::tr1::remove_reference<_Tp>::type&& __t) {
  return static_cast<_Tp&&>(__t);
}
}
#endif
