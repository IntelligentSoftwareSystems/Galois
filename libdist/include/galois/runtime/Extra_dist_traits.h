#ifndef GALOIS_RUNTIME_EXTRA_TRAITS_H
#define GALOIS_RUNTIME_EXTRA_TRAITS_H

#include <type_traits>
#include <boost/mpl/has_xxx.hpp>

//#ifndef _GALOIS_EXTRA_TRAITS_
//#define _GALOIS_EXTRA_TRAITS_
// needed when clang did not define trivially copyable, but clang 3.8 has it
//from libc++, clang specific
//namespace std {
//#ifdef __clang__
//template <class T> struct is_trivially_copyable;
//template <class _Tp> struct is_trivially_copyable
//  : public std::integral_constant<bool, __is_trivially_copyable(_Tp)>
//{};
//#else
//#if __GNUC__ < 5
//template<class T>
//using is_trivially_copyable = is_trivial<T>;
//#endif
//#endif
//}
//#endif

#if __GNUC__ < 5
#define __is_trivially_copyable(type) __has_trivial_copy(type)
#else
#define __is_trivially_copyable(type) std::is_trivially_copyable<type>::value 
#endif

namespace galois {
namespace runtime {

  BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_has_serialize)

  template<typename T>
  struct has_serialize : public has_tt_has_serialize<T> {};

  BOOST_MPL_HAS_XXX_TRAIT_DEF(tt_is_copyable)
    //! User assertion that class is trivially copyable
  template<typename T>
  struct is_copyable :  public has_tt_is_copyable<T> {};

  template<typename T>
  struct is_serializable {
      static const bool value = has_serialize<T>::value || is_copyable<T>::value || __is_trivially_copyable(T);
    };

  template<typename T>
  struct is_memory_copyable {
      static const bool value = is_copyable<T>::value || __is_trivially_copyable(T);
    };

} //namepace Runtime
} //namepace Galois

#endif
