#ifndef GALOIS_CHECKEDOBJECT_H
#define GALOIS_CHECKEDOBJECT_H

#include "galois/runtime/Context.h"

namespace galois {

/**
 * Conflict-checking wrapper for any type.  Performs global conflict detection
 * on the enclosed object.  This enables arbitrary types to be managed by the
 * Galois runtime.
 */
template<typename T>
class GChecked : public galois::runtime::Lockable {
  T val;

public:
  template<typename... Args>
  GChecked(Args&&... args): val(std::forward<Args>(args)...) { }

  T& get(galois::MethodFlag m = MethodFlag::WRITE) {
    galois::runtime::acquire(this, m);
    return val;
  }

  const T& get(galois::MethodFlag m = MethodFlag::WRITE) const {
    galois::runtime::acquire(const_cast<GChecked*>(this), m);
    return val;
  }
};

template<>
class GChecked<void>: public galois::runtime::Lockable {
public:
  void get(galois::MethodFlag m = MethodFlag::WRITE) const {
    galois::runtime::acquire(const_cast<GChecked*>(this), m);
  }
};

}

#endif // _GALOIS_CHECKEDOBJECT_H
