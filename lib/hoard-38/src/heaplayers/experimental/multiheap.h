#ifndef _MULTIHEAP_H_
#define _MULTIHEAP_H_

#if defined(WIN32) || defined(__WIN32__) || defined(_WIN32)
#ifndef WIN32
#define WIN32 1
#endif
#include <windows.h>
#include <process.h>
#endif

template <int NumHeaps, class Super>
class MultiHeap : public Super {
public:

  inline void * malloc (size_t sz) {
    // Hash the thread id.
    // We assume that it's impossible for two threads to collide.
    int tid = getThreadId() % NumHeaps;
    void * ptr = mySuper[tid].malloc (sz + sizeof(double));
    *((int *) ptr) = tid;
    void * newPtr = (void *) ((double *) ptr + 1);
    return newPtr;
  }

  inline void free (void * ptr) {
    // Return the object to its own heap.
    void * originalPtr = (void *) ((double *) ptr - 1);
    int heapIndex = *((int *) originalPtr);
    // FIX ME! A 'cache' would be nice here...
    // FIX ME! We need a lock here.
    mySuper[heapIndex].free (originalPtr);
  }

private:

  Super mySuper[NumHeaps];

  int getThreadId (void) {
#if WIN32
    return GetCurrentThreadId();
#endif
  }

};

#endif
