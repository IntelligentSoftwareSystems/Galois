/* -*- C++ -*- */

#ifndef _THREADHEAP_H_
#define _THREADHEAP_H_

#include <assert.h>
#include <new>

#if !defined(_WIN32)
#include <pthread.h>
#endif

#if defined(__SVR4) // Solaris
extern "C" unsigned int lwp_self(void);
#endif

/*

  A ThreadHeap comprises NumHeaps "per-thread" heaps.

  To pick a per-thread heap, the current thread id is hashed (mod NumHeaps).

  malloc gets memory from its hashed per-thread heap.
  free returns memory to its hashed per-thread heap.

  (This allows the per-thread heap to determine the return
  policy -- 'pure private heaps', 'private heaps with ownership',
  etc.)

  NB: We assume that the thread heaps are 'locked' as needed.  */

namespace HL {

template <int NumHeaps, class PerThreadHeap>
class ThreadHeap : public PerThreadHeap {
public:

  inline void * malloc (size_t sz) {
    int tid = getThreadId() % NumHeaps;
    assert (tid >= 0);
    assert (tid < NumHeaps);
    return getHeap(tid)->malloc (sz);
  }

  inline void free (void * ptr) {
    int tid = getThreadId() % NumHeaps;
    assert (tid >= 0);
    assert (tid < NumHeaps);
    getHeap(tid)->free (ptr);
  }

  inline size_t getSize (void * ptr) {
    int tid = getThreadId() % NumHeaps;
    assert (tid >= 0);
    assert (tid < NumHeaps);
    return getHeap(tid)->getSize (ptr);
  }

    
private:

  // Access the given heap within the buffer.
  inline PerThreadHeap * getHeap (int index) {
    assert (index >= 0);
    assert (index < NumHeaps);
    return &ptHeaps[index];
  }

  // Get a suitable thread id number.
  inline static volatile int getThreadId (void);

  PerThreadHeap ptHeaps[NumHeaps];

};

}


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
HL::ThreadHeap<NumHeaps, PerThreadHeap>::getThreadId (void) {
#if defined(WIN32)
  // It looks like thread id's are always multiples of 4, so...
  int tid = GetCurrentThreadId() >> 2;
  // Now hash in some of the first bits.
//  return (tid & ~(1024-1)) ^ tid;
  return tid;
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
  //  printf ("lwp_self = %d\n", (int) lwp_self());
  return (int) lwp_self();
#endif
#if defined(__APPLE__)
  return (int) pthread_self();
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
