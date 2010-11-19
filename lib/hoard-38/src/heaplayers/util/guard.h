// -*- C++ -*-

#ifndef _GUARD_H_
#define _GUARD_H_

namespace HL {

  template <class LockType>
  class Guard {
  public:
    inline Guard (LockType& l)
      : _lock (l)
      {
	_lock.lock();
      }

    inline ~Guard (void) {
      _lock.unlock();
    }
  private:
    LockType& _lock;
  };

}

#endif
