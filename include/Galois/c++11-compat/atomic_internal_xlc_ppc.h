#include <builtins.h>

#error "Broken"

/*
 * From: 
 *   http://www.cl.cam.ac.uk/~pes20/cpp/cpp0xmappings.html 
 *     and 
 *   Batty et al. Clarifying and Compiling C/C++ Concurrency: from C++11 to POWER. POPL 2011.
 *     (http://www.cl.cam.ac.uk/~pes20/cppppc/)
 */

namespace detail {

inline bool atomic_compare_exchange_strong32(volatile int* __a, int* __e, int* __d, std::memory_order _succ, std::memory_order _fail) {
  bool tmp;
  int v = *__e;
  switch (_succ) {
    case std::memory_order_relaxed: return __compare_and_swap(__a, &v, *__d);
    case std::memory_order_consume: abort();
    case std::memory_order_acquire: tmp = __compare_and_swap(__a, &v, *__d); __isync(); return tmp;
    case std::memory_order_release: __lwsync(); return __compare_and_swap(__a, &v, *__d);
    case std::memory_order_acq_rel: __lwsync(); tmp = __compare_and_swap(__a, &v, *__d); __isync(); return tmp;
    case std::memory_order_seq_cst: __sync(); tmp = __compare_and_swap(__a, &v, *__d); __isync(); return tmp;
    default: abort();
  }
  // v contains old value in __a;
  return tmp;
}

#ifdef __PPC64__
inline bool atomic_compare_exchange_strong64(volatile long* __a, long* __e, long* __d, std::memory_order _succ, std::memory_order _fail) {
  bool tmp;
  long v = *__e;
  switch (_succ) {
    case std::memory_order_relaxed: return __compare_and_swaplp(__a, &v, *__d);
    case std::memory_order_consume: abort();
    case std::memory_order_acquire: tmp = __compare_and_swaplp(__a, &v, *__d); __isync(); return tmp;
    case std::memory_order_release: __lwsync(); return __compare_and_swaplp(__a, &v, *__d);
    case std::memory_order_acq_rel: __lwsync(); tmp = __compare_and_swaplp(__a, &v, *__d); __isync(); return tmp;
    case std::memory_order_seq_cst: __sync(); tmp = __compare_and_swaplp(__a, &v, *__d); __isync(); return tmp;
    default: abort();
  }
  // v contains old value in __a;
  return tmp;
}
#endif

template<class _Tp>
bool atomic_compare_exchange_strong(volatile _Tp* __a, _Tp* __e, _Tp* __d, std::memory_order _succ, std::memory_order _fail) {
  // __sync_XXX gcc-type intrinsics issue a full barrier so implement using
  // lower level intrinsics
#ifdef __PPC64__
  static_assert(sizeof(_Tp) <= 8, "Operation undefined on larger types");
#else
  static_assert(sizeof(_Tp) <= 4, "Operation undefined on larger types");
#endif
  if (sizeof(_Tp) <= 4)
    return detail::atomic_compare_exchange_strong32(reinterpret_cast<volatile int*>(__a), reinterpret_cast<int*>(__e), reinterpret_cast<int*>(__d), _succ, _fail);
#ifdef __PPC64__
  else
    return detail::atomic_compare_exchange_strong64(reinterpret_cast<volatile long*>(__a), reinterpret_cast<long*>(__e), reinterpret_cast<long*>(__d), _succ, _fail);
#endif
  abort();
  return false;
}

/* 
 * Weak fence (cmp; bc; isync) which depends on PowerPC guaranteeing that
 * loads on which a branch condition (bc) instruction depends are completed
 * before and any stores happening after.
 *
 * See:
 *  http://www.rdrop.com/users/paulmck/scalability/paper/N2745r.2011.03.04a.html
 */
template<class _Tp>
void weak_fence(volatile _Tp* __a) {
  // TODO: implement this in asm
  while (*__a != *__a)
    ;
  __lwsync();
}

} // end detail

template<class _Tp>
void __atomic_store(volatile _Tp* __a, _Tp* __i, std::memory_order _m) {
  switch (_m) {
    case std::memory_order_relaxed: *__a = *__i; break;
    case std::memory_order_consume:
    case std::memory_order_acquire: abort(); break;
    case std::memory_order_release:
    case std::memory_order_acq_rel: __lwsync(); *__a = *__i;
    case std::memory_order_seq_cst: __sync(); *__a = *__i; break;
    default: abort();
  }
}

template<class _Tp>
void __atomic_load(volatile _Tp* __a, _Tp* __i, std::memory_order _m) {
  switch (_m) {
    case std::memory_order_relaxed: *__i = *__a; break;
    case std::memory_order_consume: 
    case std::memory_order_acquire: *__i = *__a; detail::weak_fence(__i); break;
    case std::memory_order_release: abort(); break;
    case std::memory_order_acq_rel: *__i = *__a; detail::weak_fence(__i); break;
    case std::memory_order_seq_cst: __sync(); *__i = *__a; detail::weak_fence(__i); break;
    default: abort();
  }
}

template<class _Tp>
void __atomic_load(volatile const _Tp* __a, _Tp* __i, std::memory_order _m) {
  __atomic_load(const_cast<_Tp*>(__a), __i, _m);
}

template<class _Tp>
bool __atomic_compare_exchange(volatile _Tp* __a, _Tp* __e, _Tp* __d, bool _weak, std::memory_order _succ, std::memory_order _fail) {
  return detail::atomic_compare_exchange_strong(__a, __e, __d, _succ, _fail);
}

template<class _Tp>
_Tp __atomic_fetch_xor(volatile _Tp* __a, _Tp __i, std::memory_order _m) {
  _Tp old;
  _Tp newval;
  do {
    old = *__a;
    newval = old ^ __i;
  } while (!__atomic_compare_exchange(__a, &old, &newval, true, _m, _m));
  return old;
}

template<class _Tp>
_Tp __atomic_fetch_add(volatile _Tp* __a, _Tp __i, std::memory_order _m) {
  _Tp old;
  _Tp newval;
  do {
    old = *__a;
    newval = old + __i;
  } while (!__atomic_compare_exchange(__a, &old, &newval, true, _m, _m));
  return old;
}
