/* -*- C++ -*- */

#ifndef _SIZETHREADHEAP_H_
#define _SIZETHREADHEAP_H_

#include <assert.h>

template <class Super>
class SizeThreadHeap : public Super {
public:
  
  inline void * malloc (size_t sz) {
    // Add room for a size field & a thread field.
	// Both of these must fit in a double.
	assert (sizeof(st) <= sizeof(double));
    st * ptr = (st *) Super::malloc (sz + sizeof(double));
    // Store the requested size.
    ptr->size = sz;
	assert (getOrigPtr(ptr + 1) == ptr);
    return (void *) (ptr + 1);
  }
  
  inline void free (void * ptr) {
    void * origPtr = (void *) getOrigPtr(ptr);
    Super::free (origPtr);
  }

  static inline volatile int getThreadId (void);

  static inline size_t& size (void * ptr) {
		return getOrigPtr(ptr)->size;
  }

  static inline int& thread (void * ptr) {
		return getOrigPtr(ptr)->tid;
	}

private:

	typedef struct _st {
		size_t size;
		int tid;
	} st;

	static inline st * getOrigPtr (void * ptr) {
		return (st *) ((double *) ptr - 1);
	}

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

template <class SuperHeap>
inline volatile int
SizeThreadHeap<SuperHeap>::getThreadId (void) {
#if defined(WIN32)
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
#if defined(POSIX) // FIX ME??
  return (int) pthread_self();
#endif
#if USE_SPROC // FIX ME
  // This hairiness has the same effect as calling getpid(),
  // but it's MUCH faster since it avoids making a system call
  // and just accesses the sproc-local data directly.
  int pid = (int) PRDA->sys_prda.prda_sys.t_pid;
  return pid;
#endif
}
  


#endif
