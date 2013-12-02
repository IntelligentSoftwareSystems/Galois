#ifndef GALOIS_C__11_COMPAT_MUTEX_H
#define GALOIS_C__11_COMPAT_MUTEX_H

#include "tuple.h"

namespace std {
template<typename _Mutex>
class lock_guard {
public:
  typedef _Mutex mutex_type;
  explicit lock_guard(mutex_type& __m): _M_device(__m) { _M_device.lock(); }
  ~lock_guard() { _M_device.unlock(); }

private:
  lock_guard(const lock_guard&);
  lock_guard& operator=(const lock_guard&);

  mutex_type& _M_device;
};

template<int _Idx>
struct __unlock_impl {
  template<typename... _Lock>
  static void __do_unlock(tuple<_Lock&...>& __locks) {
      std::get<_Idx>(__locks).unlock();
      __unlock_impl<_Idx - 1>::__do_unlock(__locks);
  }
};

template<>
struct __unlock_impl<-1> {
  template<typename... _Lock>
  static void __do_unlock(tuple<_Lock&...>&) { }
};

template<int _Idx, bool _Continue = true>
struct __try_lock_impl {
  template<typename... _Lock>
  static int __do_try_lock(tuple<_Lock&...>& __locks) {
    if (std::get<_Idx>(__locks).try_lock()) {
      return __try_lock_impl<_Idx + 1,
        _Idx + 2 < sizeof...(_Lock)>::__do_try_lock(__locks);
    } else {
      __unlock_impl<_Idx>::__do_unlock(__locks);
      return _Idx;
    }
  }
};

template<int _Idx>
struct __try_lock_impl<_Idx, false> {
  template<typename... _Lock>
  static int __do_try_lock(tuple<_Lock&...>& __locks) {
    if (std::get<_Idx>(__locks).try_lock())
      return -1;
    else {
      __unlock_impl<_Idx>::__do_unlock(__locks);
      return _Idx;
    }
  }
};


template<typename _Lock1, typename _Lock2, typename... _Lock3>
int try_lock(_Lock1& __l1, _Lock2& __l2, _Lock3&... __l3) {
  tuple<_Lock1&, _Lock2&, _Lock3&...> __locks(__l1, __l2, __l3...);
  return __try_lock_impl<0>::__do_try_lock(__locks);
}
}
#endif
