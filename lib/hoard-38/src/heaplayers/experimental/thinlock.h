// -*- C++ -*-

#ifndef _THINLOCK_H_
#define _THINLOCK_H_

#include <windows.h>

class ThinLock {
public:

  ThinLock (void)
  {
    InitializeCriticalSection (&crit);
  }

  inline void lock (void) {
    EnterCriticalSection (&crit);
  }

  inline void unlock (void) {
    LeaveCriticalSection (&crit);
  }

private:
  CRITICAL_SECTION crit;
};


#endif
