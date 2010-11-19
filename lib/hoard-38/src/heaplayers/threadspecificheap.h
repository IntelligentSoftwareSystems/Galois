/* -*- C++ -*- */

#ifndef _THREADHEAP_H_
#define _THREADHEAP_H_

#include <assert.h>
#include <new.h>

/*

  A ThreadHeap comprises NumHeaps "per-thread" heaps.

  To pick a per-thread heap, the current thread id is hashed (mod NumHeaps).

  malloc gets memory from its hashed per-thread heap.
  free returns memory to its hashed per-thread heap.

  (This allows the per-thread heap to determine the return
  policy -- 'pure private heaps', 'private heaps with ownership',
  etc.)

  NB: We assume that the thread heaps are 'locked' as needed.  */


template <int NumHeaps, class PerThreadHeap>
class ThreadHeap : public PerThreadHeap {
public:

  ThreadHeap (void)
  {
  }

  inline void * malloc (size_t sz) {
    int tid = getThreadId() % NumHeaps;
    return getHeap(tid)->malloc (sz);
  }

  inline void free (void * ptr) {
    int tid = getThreadId() % NumHeaps;
    getHeap(tid)->free (ptr);
  }


private:

  // Access the given heap within the buffer.
  PerThreadHeap * getHeap (int index) {
    assert (index >= 0);
    assert (index < NumHeaps);
	return &ptHeaps[index];
  }

  // Get a suitable thread id number.
  inline static volatile int getThreadId (void);

  PerThreadHeap ptHeaps[NumHeaps];

};


// A platform-dependent way to get a thread id.

// Include the necessary platform-dependent crud.
#if defined(WIN32) || defined(__WIN32__) || defined(_WIN32)
#ifndef WIN32
#define WIN32 1
#endif
#include <windows.h>
#include <process.h>
#endif

template <int NumHeaps, class PerThreadHeap>
inline volatile int
ThreadHeap<NumHeaps, PerThreadHeap>::getThreadId (void) {
#if WIN32
  return GetCurrentThreadId();
#endif
#if defined(__BEOS__)
  return find_thread(0);
#endif
#if defined(__linux)
  // Consecutive thread id's in Linux are 1024 apart;
  // dividing off the 1024 gives us an appropriate thread id.
  return (int) pthread_self() >> 10; // (>> 10 = / 1024)
#endif
#if defined(__SVR4)
  return (int) lwp_self();
#endif
#if defined(POSIX)
  return (int) pthread_self();
#endif
#if USE_SPROC
  // This hairiness has the same effect as calling getpid(),
  // but it's MUCH faster since it avoids making a system call
  // and just accesses the sproc-local data directly.
  int pid = (int) PRDA->sys_prda.prda_sys.t_pid;
  return pid;
#endif
}
  

#endif
