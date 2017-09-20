#ifndef _ATOMIC_WRAPPER_H_
#define _ATOMIC_WRAPPER_H_

#include <atomic>
#include <type_traits>

#ifndef _GALOIS_EXTRA_TRAITS_
#define _GALOIS_EXTRA_TRAITS_
//from libc++, clang specific
namespace std {
#ifdef __clang__
// not required for clang 3.8.0 : may be required for older versions 
//template <class T> struct is_trivially_copyable;
//template <class _Tp> struct is_trivially_copyable
//  : public std::integral_constant<bool, __is_trivially_copyable(_Tp)>
//{};
#else
#if __GNUC__ < 5
template<class T>
using is_trivially_copyable = is_trivial<T>;
#endif
#endif
}
#endif
//#define __is_trivially_copyable(type)  __has_trivial_copy(type)


namespace galois {

  template<class T>
    class CopyableAtomic : public std::atomic<T>
  {
    public:

      CopyableAtomic() : 
        std::atomic<T>(T{})
        {}

      constexpr CopyableAtomic(T desired) : 
        std::atomic<T>(desired) 
    {}

      constexpr CopyableAtomic(const CopyableAtomic<T>& other) :
        CopyableAtomic(other.load(std::memory_order_relaxed))
    {}

      CopyableAtomic& operator=(const CopyableAtomic<T>& other) {
        this->store(other.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
      }

      // only typedef if T is trivially copyable to use mem copy in serialize/deserialize.
      typedef typename std::enable_if<std::is_trivially_copyable<T>::value, int>::type tt_is_copyable;
  };
}
#endif
