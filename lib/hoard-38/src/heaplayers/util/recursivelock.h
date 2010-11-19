/* -*- C++ -*- */

#ifndef _RECURSIVELOCK_H_
#define _RECURSIVELOCK_H_

/**
 * @class RecursiveLockType
 * @brief Implements a recursive lock using some base lock representation.
 * @param BaseLock The base lock representation.
 */

namespace HL {

template <class BaseLock>
class RecursiveLockType : public BaseLock {
public:

  inline RecursiveLockType (void);

  inline void lock (void);
  inline void unlock (void);

private:
  int tid;	/// The lock owner's thread id. -1 if unlocked.
  int count;	/// The recursion depth of the lock.
};


};

template <class BaseLock>
HL::RecursiveLockType<BaseLock>::RecursiveLockType (void)
  : tid (-1),
    count (0)
{}

template <class BaseLock>
void HL::RecursiveLockType<BaseLock>::lock (void) {
  int currthread = GetCurrentThreadId();
  if (tid == currthread) {
    count++;
  } else {
    BaseLock::lock();
    tid = currthread;
    count++;
  }
}

template <class BaseLock>
void HL::RecursiveLockType<BaseLock>::unlock (void) {
  int currthread = GetCurrentThreadId();
  if (tid == currthread) {
    count--;
    if (count == 0) {
      tid = -1;
      BaseLock::unlock();
    }
  } else {
    // We tried to unlock it but we didn't lock it!
    // This should never happen.
    assert (0);
    abort();
  }
}

#endif
