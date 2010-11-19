/* -*- C++ -*- */

#ifndef _WINLOCK_H_
#define _WINLOCK_H_

#if defined(_WIN32)

#include <windows.h>

/**
 * @class WinLockType
 * @brief Locking using Win32 mutexes.
 *
 * Note that this lock type detects whether we are running on a
 * multiprocessor.  If not, then we do not use atomic operations.
 */

#if 1

namespace HL {

class WinLockType {
public:

  WinLockType (void)
    : mutex (0)
  {}

  ~WinLockType (void)
  {
    mutex = 0;
  }

  inline void lock (void) {
#if 1
    // We're on a multiprocessor - use atomic operations.
    while (InterlockedExchange ((long *) &mutex, 1) != 0) 
      Sleep (0);
#endif
  }

  inline void unlock (void) {
    mutex = 0;
    // InterlockedExchange (&mutex, 0);
  }

private:
  unsigned int mutex;
  bool onMultiprocessor (void) {
    SYSTEM_INFO infoReturn[1];
    GetSystemInfo (infoReturn);
    if (infoReturn->dwNumberOfProcessors == 1) {
      return FALSE;
    } else {
      return TRUE;
    }
  }
};

};

#else

#include <windows.h>

class WinLockType {
public:

  WinLockType (void)
  {
    InitializeCriticalSection(&mutex);
  }

  ~WinLockType (void)
  {
    DeleteCriticalSection(&mutex);
  }

  inline void lock (void) {
    EnterCriticalSection(&mutex);
  }

  inline void unlock (void) {
    LeaveCriticalSection(&mutex);
  }

private:
  CRITICAL_SECTION mutex;
};

#endif

#endif

#endif
