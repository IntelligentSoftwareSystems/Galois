namespace detail {

template<class _Tp>
bool atomic_compare_exchange_strong(volatile _Tp* __a, _Tp* __e, _Tp* __d, std::memory_order _succ, std::memory_order _fail) {
  static_assert(sizeof(_Tp) <= 8, "Operation undefined on larger types");
  return __sync_bool_compare_and_swap(__a, *__e, *__d);
}

} // end detail

template<class _Tp>
void __atomic_store(volatile _Tp* __a, _Tp* __i, std::memory_order _m) {
  __sync_synchronize();
  *__a = *__i;
  __sync_synchronize();
}

template<class _Tp>
void __atomic_load(volatile _Tp* __a, _Tp* __i, std::memory_order _m) {
  __sync_synchronize();
  *__i = *__a;
  __sync_synchronize();
}

template<class _Tp>
void __atomic_load(volatile const _Tp* __a, _Tp* __i, std::memory_order _m) {
  __sync_synchronize();
  *__i = *__a;
  __sync_synchronize();
}

template<class _Tp>
bool __atomic_compare_exchange(volatile _Tp* __a, _Tp* __e, _Tp* __d, bool _weak, std::memory_order _succ, std::memory_order _fail) {
  return detail::atomic_compare_exchange_strong(__a, __e, __d, _succ, _fail);
}

template<class _Tp>
_Tp __atomic_fetch_xor(volatile _Tp* __a, _Tp __i, std::memory_order _m) {
  return __sync_fetch_and_xor(__a, __i);
}

template<class _Tp>
_Tp __atomic_fetch_add(volatile _Tp* __a, _Tp __i, std::memory_order _m) {
  return __sync_fetch_and_add(__a, __i);
}
