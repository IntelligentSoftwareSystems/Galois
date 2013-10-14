#ifndef GALOIS_C__11_COMPAT_ATOMIC_H
#define GALOIS_C__11_COMPAT_ATOMIC_H

#include "type_traits.h"


namespace std {

typedef enum memory_order
  {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
  } memory_order;

}

#if __IBMCPP__ && __PPC__ 
//# include "atomic_internal_xlc_ppc.h"
# include "atomic_internal_gcc_generic.h"
#elif __GNUC__
# include "atomic_internal_gcc_generic.h"
#else
# error "Unknown machine architecture"
#endif

namespace std {

template<class _Tp>
class atomic {
  _Tp _M_i;

  atomic(const atomic&);
  atomic& operator=(const atomic&);
  atomic& operator=(const atomic&) volatile;

public:
  atomic() { }
  constexpr atomic(_Tp __i): _M_i(__i) { }
  operator _Tp() const { return load(); }
  operator _Tp() const volatile { return load(); }
  _Tp operator=(_Tp __i) { store(__i); return __i; }
  _Tp operator=(_Tp __i) volatile { store(__i); return __i; }

  void store(_Tp __i, memory_order _m = memory_order_seq_cst) { __atomic_store(&_M_i, &__i, _m); }
  void store(_Tp __i, memory_order _m = memory_order_seq_cst) volatile { __atomic_store(&_M_i, &__i, _m); }
  _Tp load(memory_order _m = memory_order_seq_cst) const {
    _Tp tmp;
    __atomic_load(&_M_i, &tmp, _m);
    return tmp;
  }
  _Tp load(memory_order _m = memory_order_seq_cst) const volatile {
    _Tp tmp;
    __atomic_load(&_M_i, &tmp, _m);
    return tmp;
  }
  _Tp exchange(_Tp __i, memory_order _m = memory_order_seq_cst) { 
    return __atomic_exchange(&_M_i, __i, _m);
  }
  _Tp exchange(_Tp __i, memory_order _m = memory_order_seq_cst) volatile { 
    return __atomic_exchange(&_M_i, __i, _m);
  }
  bool compare_exchange_weak(_Tp& __e, _Tp __i, memory_order _m1, memory_order _m2) {
    return __atomic_compare_exchange(&_M_i, &__e, &__i, true, _m1, _m2);
  }
  bool compare_exchange_weak(_Tp& __e, _Tp __i, memory_order _m1, memory_order _m2) volatile {
    return __atomic_compare_exchange(&_M_i, &__e, &__i, true, _m1, _m2);
  }
  bool compare_exchange_weak(_Tp& __e, _Tp __i, memory_order _m = memory_order_seq_cst) {
    return compare_exchange_weak(__e, __i, _m, _m);
  }
  bool compare_exchange_weak(_Tp& __e, _Tp __i, memory_order _m = memory_order_seq_cst) volatile {
    return compare_exchange_weak(__e, __i, _m, _m);
  }
  bool compare_exchange_strong(_Tp& __e, _Tp __i, memory_order _m1, memory_order _m2) {
    return __atomic_compare_exchange(&_M_i, &__e, &__i, false, _m1, _m2);
  }
  bool compare_exchange_strong(_Tp& __e, _Tp __i, memory_order _m1, memory_order _m2) volatile {
    return __atomic_compare_exchange(&_M_i, &__e, &__i, false, _m1, _m2);
  }
  bool compare_exchange_strong(_Tp& __e, _Tp __i, memory_order _m = memory_order_seq_cst) {
    return compare_exchange_strong(__e, __i, _m, _m);
  }
  bool compare_exchange_strong(_Tp& __e, _Tp __i, memory_order _m = memory_order_seq_cst) volatile {
    return compare_exchange_strong(__e, __i, _m, _m);
  }

  template<bool Enable = std::is_integral<_Tp>::value>
  _Tp fetch_xor(_Tp __i, memory_order _m = memory_order_seq_cst, typename std::enable_if<Enable>::type* = 0) {
    return __atomic_fetch_xor(&_M_i, __i, _m);
  }
  template<bool Enable = std::is_integral<_Tp>::value>
  _Tp fetch_xor(_Tp __i, memory_order _m = memory_order_seq_cst, typename std::enable_if<Enable>::type* = 0) volatile {
    return __atomic_fetch_xor(&_M_i, __i, _m);
  }

  template<bool Enable = std::is_integral<_Tp>::value>
  _Tp fetch_or(_Tp __i, memory_order _m = memory_order_seq_cst, typename std::enable_if<Enable>::type* = 0) {
    return __atomic_fetch_or(&_M_i, __i, _m);
  }
  template<bool Enable = std::is_integral<_Tp>::value>
  _Tp fetch_or(_Tp __i, memory_order _m = memory_order_seq_cst, typename std::enable_if<Enable>::type* = 0) volatile {
    return __atomic_fetch_or(&_M_i, __i, _m);
  }

  template<bool Enable = std::is_integral<_Tp>::value>
  _Tp fetch_add(_Tp __i, memory_order _m = memory_order_seq_cst, typename std::enable_if<Enable>::type* = 0) {
    return __atomic_fetch_add(&_M_i, __i, _m);
  }
  template<bool Enable = std::is_integral<_Tp>::value>
  _Tp operator++() {
    return fetch_add(1) + 1;
  }
  template<bool Enable = std::is_integral<_Tp>::value>
  _Tp fetch_add(_Tp __i, memory_order _m = memory_order_seq_cst, typename std::enable_if<Enable>::type* = 0) volatile {
    return __atomic_fetch_add(&_M_i, __i, _m);
  }
  template<bool Enable = std::is_integral<_Tp>::value>
  _Tp operator++() volatile {
    return fetch_add(1) + 1;
  }
};

}

#endif
