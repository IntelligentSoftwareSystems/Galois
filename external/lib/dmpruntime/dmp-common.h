// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// The interface common to both the runtime and user code
// This includes a wrapper around some PThread functions.
//

#ifndef _DMP_COMMON_H_
#define _DMP_COMMON_H_

#include <stdint.h>
#include <pthread.h>

#include "config.h"
#include "dmp-common-config.h"

//--------------------------------------------------------------
// Compiler-inserted hooks
// Implemented in runtime-model-*.cpp
//--------------------------------------------------------------

void DMPcommit(int chunkSize);
void DMPprepareForExternalCall(int chunkSize);
void DMPprepareForIndirectCall(int chunkSize, void* fnaddr);
void DMPmembarrier(void);          // user memory barrier (send to next quantum)
void DMPmembarrierResource(void);  // resource memory barrier (send to serial mode)

// for debugging (see runtime-profiling.cpp)
void DMPsetCodeLocation(int loc);

#ifdef DMP_ENABLE_BUFFERED_MODE

// Contained accesses.
// The compiler verifies these are contained within one MOT entry.
void DMPloadBufferedContained(void* addr, size_t size, void* outbuffer);
void DMPstoreBufferedContained(void* addr, size_t size, void* inbuffer);

// Uncontained accesses.
void DMPloadBufferedRange(void* addr, size_t size, void* outbuffer);
void DMPstoreBufferedRange(void* addr, size_t size, void* inbuffer);

// Allocated memory: must remove from the write buffer when deallocated!
void DMPremoveBufferedRange(void* addr, size_t size);

// Type specializations
// Note that Int8* can never be an uncontained access.
uint8_t  DMPloadBufferedContainedInt8  (uint8_t*  addr);
uint16_t DMPloadBufferedContainedInt16 (uint16_t* addr);
uint32_t DMPloadBufferedContainedInt32 (uint32_t* addr);
uint64_t DMPloadBufferedContainedInt64 (uint64_t* addr);
float    DMPloadBufferedContainedFloat (float*    addr);
double   DMPloadBufferedContainedDouble(double*   addr);
void*    DMPloadBufferedContainedPtr   (void**    addr);

uint16_t DMPloadBufferedRangeInt16 (uint16_t* addr);
uint32_t DMPloadBufferedRangeInt32 (uint32_t* addr);
uint64_t DMPloadBufferedRangeInt64 (uint64_t* addr);
float    DMPloadBufferedRangeFloat (float*    addr);
double   DMPloadBufferedRangeDouble(double*   addr);
void*    DMPloadBufferedRangePtr   (void**    addr);

void DMPstoreBufferedContainedInt8  (uint8_t*  addr, uint8_t  value);
void DMPstoreBufferedContainedInt16 (uint16_t* addr, uint16_t value);
void DMPstoreBufferedContainedInt32 (uint32_t* addr, uint32_t value);
void DMPstoreBufferedContainedInt64 (uint64_t* addr, uint64_t value);
void DMPstoreBufferedContainedFloat (float*    addr, float    value);
void DMPstoreBufferedContainedDouble(double*   addr, double   value);
void DMPstoreBufferedContainedPtr   (void**    addr, void*    value);

void DMPstoreBufferedRangeInt16 (uint16_t* addr, uint16_t value);
void DMPstoreBufferedRangeInt32 (uint32_t* addr, uint32_t value);
void DMPstoreBufferedRangeInt64 (uint64_t* addr, uint64_t value);
void DMPstoreBufferedRangeFloat (float*    addr, float    value);
void DMPstoreBufferedRangeDouble(double*   addr, double   value);
void DMPstoreBufferedRangePtr   (void**    addr, void*    value);

#else

// Contained accesses.
// The compiler verifies these are contained within one MOT entry.
void DMPloadContained(void* addr);
void DMPstoreContained(void* addr);

// Uncontained accesses.
void DMPloadRange(void* addr, size_t size);
void DMPstoreRange(void* addr, size_t size);

#endif

//--------------------------------------------------------------
// LibC stubs
//--------------------------------------------------------------

void DMPmemcpy(void* dst, const void* src, size_t size);
void DMPmemset(void* addr, int val, size_t size);

// TODO
// Implement LibC functions used by parsec/splash benchmarks.
// Here's a list:
//   mem{set,cpy,move,chr,rchr,cmp}
//   strn{cmp,casecmp}
//   ato{i,f}
//   str{chr,cmp,cpy,len,pbrk,str,tod,tol}
//   s{n}printf
//
// Custom implementations would have two advantages:
//   1) More accurate logical-time accounting.
//      e.g., we could adjust logical-time based on the buffer size
//   2) The functions would execute in parallel mode.
//      Currently, they're forced to serial mode by DMPprepareForExternalCall().
//
// For O|S, custom implementations are easy for the functions
// that take a sized buffer, and slightly harder for functions
// that take a string.  For example, memcpy() might look like:
//
//   void* DMPmemcpy(void* a, void* b, size_t s) {
//     DMPstoreRange(a, s);
//     DMPloadRange(b, s);
//     return memcpy(a, b, s);
//   }
//
// For buffering modes, things get harder because we'd need to
// add a "copy" primitive to the write-buffer.
//
// There is another difficulty: for C++ code, these LibC functions
// are defined in the std:: namespace, so we'd have to do something
// more clever than just:
//
//   #define DMPmemcpy memcpy
//

//--------------------------------------------------------------
// Pthreads
//--------------------------------------------------------------

typedef struct DMPonce DMPonce;
typedef struct DMPmutex DMPmutex;
typedef struct DMPrwlock DMPrwlock;
typedef struct DMPcondvar DMPcondvar;
typedef struct DMPbarrier DMPbarrier;
typedef struct DMPsemaphore DMPsemaphore;

#include "dmp-common-resource.h"

// runtime-main.cpp
int DMPmain(int argc, char** argv);

// library-thread.cpp
int DMPthread_create(pthread_t*, pthread_attr_t*,
                     void *(*start_routine)(void*), void *arg);
int DMPthread_join(pthread_t thread, void **value_ptr);
int DMPthread_cancel(pthread_t thread);

// library-once.cpp
struct DMPonce {
  int done;
  DMPresource resource;
};
#define DMP_THREAD_ONCE_INITIALIZER  {0,DMP_RESOURCE_ONCE_INIT}
int DMPonce_once(DMPonce* once, void (*init_routine)(void));

// library-mutex.cpp
struct DMPmutex {
  int spinlock;
  DMPresource resource;
};
#define DMP_THREAD_MUTEX_INITIALIZER  {0,DMP_RESOURCE_MUTEX_INIT}
int DMPmutex_init(DMPmutex* mutex, void* attr);
int DMPmutex_destroy(DMPmutex* mutex);
int DMPmutex_lock(DMPmutex* mutex);
int DMPmutex_trylock(DMPmutex* mutex);
int DMPmutex_unlock(DMPmutex* mutex);

// library-rwlock.cpp
struct DMPrwlock {
  int state;           // 0 = free, -1 = owned by a writer, >1 = number of readers
  int waiting_writers; // number of waiting writers, to prevent writer starvation
  DMPresource resource;
};
#define DMP_THREAD_RWLOCK_INITIALIZER  {0,0,DMP_RESOURCE_RWLOCK_INIT}
int DMPrwlock_init(DMPrwlock* rwlock, void* attr);
int DMPrwlock_destroy(DMPrwlock* rwlock );
int DMPrwlock_rdlock(DMPrwlock* rwlock);
int DMPrwlock_wrlock(DMPrwlock* rwlock);
int DMPrwlock_tryrdlock(DMPrwlock* rwlock);
int DMPrwlock_trywrlock(DMPrwlock* rwlock);
int DMPrwlock_unlock(DMPrwlock* rwlock);

// library-cv.cpp
struct DMPcondvar {
  DMPwaiter* first;
  DMPresource resource;
};
#define DMP_THREAD_COND_INITIALIZER  {NULL,DMP_RESOURCE_CONDVAR_INIT}
int DMPcondvar_init(DMPcondvar* C, void* attr);
int DMPcondvar_destroy(DMPcondvar* C);
int DMPcondvar_wait(DMPcondvar* C, DMPmutex* lock);
int DMPcondvar_signal(DMPcondvar* C);
int DMPcondvar_broadcast(DMPcondvar* C);

// library-barrier.cpp
struct DMPbarrier {
  unsigned needed;
  unsigned arrived;
  DMPwaiter* first;
  DMPresource resource;
};
int DMPbarrier_init(DMPbarrier *B, void* attr, unsigned needed);
int DMPbarrier_destroy(DMPbarrier *B);
int DMPbarrier_wait(DMPbarrier *B);
int DMPbarrier_init_splash(DMPbarrier *B);
int DMPbarrier_wait_splash(DMPbarrier *B, unsigned needed);

// library-semaphore.cpp
struct DMPsemaphore {
  unsigned count;
  DMPresource resource;
};
int DMPsemaphore_init(DMPsemaphore* sem, int pshared, unsigned value);
int DMPsemaphore_wait(DMPsemaphore* sem);
int DMPsemaphore_post(DMPsemaphore* sem);

#endif  // _DMP_COMMON_H_
