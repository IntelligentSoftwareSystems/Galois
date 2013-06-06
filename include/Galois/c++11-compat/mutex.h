#ifndef GALOIS_C__11_COMPAT_MUTEX_H
#define GALOIS_C__11_COMPAT_MUTEX_H

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
}
#endif
