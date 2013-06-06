#ifndef GALOIS_C__11_COMPAT_ALGORITHM_H
#define GALOIS_C__11_COMPAT_ALGORITHM_H

#include <algorithm>

#include <boost/tr1/type_traits.hpp>

namespace std {
template<typename _Tp>
constexpr typename std::tr1::remove_reference<_Tp>::type&&
move(_Tp&& __t) { 
  return static_cast<typename std::tr1::remove_reference<_Tp>::type&&>(__t); 
}
}
#endif
