// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// The interface used only by the runtime, and not exported to user code.
//

#ifndef _DMP_INTERNAL_H_
#define _DMP_INTERNAL_H_

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <memory.h>
#include <pthread.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>

#ifdef __linux__
typedef uint32_t u32; // needed on RedHat for some reason
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
static inline int futex(volatile int* uaddr, int op, int val,
                        const struct timespec* timeout, int* uaddr2, int val2) {
  return syscall(SYS_futex, uaddr, op, val, timeout, uaddr2, val2);
}
#endif

#define ROUNDUP_MOD_N(x, n) (((x) % (n) != 0) ? ((x) / (n)) * (n) + (n) : (x))

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

// Generate a compile-time error if 'e' is false.
#define dmp_static_assert(e) ((void)sizeof(char[1 - 2 * !(e)]))

// Verbose DMP asserts
#ifndef NDEBUG
#define DMP_ASSERT(e)                                                          \
  do {                                                                         \
    if (!(e))                                                                  \
      DMPprintScheduler();                                                     \
    assert(e);                                                                 \
  } while (0)
#else
#define DMP_ASSERT(e)
#endif

// Cacheline size in bytes.
// This is a #define since LLVM doesn't consider 'const' variables 'const'
// exprs.
#define CACHELINE_SIZE 64

// Alignment
#define CACHELINE_ALIGNED __attribute__((aligned(CACHELINE_SIZE)))
#define CACHELINE_PADDED(type, var)                                            \
  union CACHELINE_ALIGNED {                                                    \
    type var;                                                                  \
    char __pad_##var[CACHELINE_SIZE];                                          \
  }

// Branch prediction!
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// Inlining notes:
// -- Fast-path checks are done in DMPcommit/load/store
// -- Slow-path checks are done in a second non-inlined function
#define FORCE_INLINE __attribute__((always_inline))

//--------------------------------------------------------------
// C API
//--------------------------------------------------------------

extern "C" {

#include "debug.h"
#include "dmp-common.h"
}

extern int (*real_posix_memalign)(void**, size_t, size_t);
extern void* (*real_malloc)(size_t);
extern void* (*real_realloc)(void*, size_t);
extern void (*real_free)(void*);
void* alloc_cache_aligned(size_t size);

//--------------------------------------------------------------
// Atomic internals
//--------------------------------------------------------------

// Volatile here prevents register allocation of variables of this type.
// Appropriate memory barriers and atomic ops (of the "__sync" variety)
// are required for true atomic access, of course.
typedef volatile bool atomic_bool_t;
typedef volatile int32_t atomic_int_t;
typedef volatile uint32_t atomic_uint_t;
typedef volatile uint64_t atomic_uint64_t;

// Volatile ops
#define VOLATILE_LOAD(x) (*(volatile __typeof__(x)*)&(x))
#define VOLATILE_STORE(x, y) ((*(volatile __typeof__(x)*)&(x)) = (y))

// Spinlocks
// Any integer location can be treaded as a spinlock.
// Trylock returns non-zero on success.
//
// Lock operations emit an acquire memory barrier:
//   * Previous stores may not be visible after this operation
//   * Previous loads may not be complete after this operation
//   * No loads or stores may be reordered to before this operation
//
// Unlock emits a release memory barrier:
//   * Previous stores will be visible before this operation
//   * Previous loads will be visible before this operation
//   * Loads may be reordered to before this operation

#define DMP_SPINLOCK_TRYLOCK(l) (__sync_lock_test_and_set((l), 1) == 0)

#define DMP_SPINLOCK_LOCK(l)                                                   \
  do {                                                                         \
  } while (!DMP_SPINLOCK_TRYLOCK(l))

#define DMP_SPINLOCK_UNLOCK(l) (__sync_lock_release(l))

//--------------------------------------------------------------
// OS-specific
//--------------------------------------------------------------

// linux
#ifdef __linux__
#define YIELD_CPU() pthread_yield()
#endif

// OS X
#ifdef __APPLE_CC__
#ifdef __MACH__
#define YIELD_CPU() pthread_yield_np()
#endif
#endif

// busy-wait
#define YIELD()                                                                \
  {                                                                            \
    if (DMPnumRunnableThreads > DMP_NUM_PHYSICAL_PROCESSORS) {                 \
      YIELD_CPU();                                                             \
    }                                                                          \
  }

// futex-wait
// The futex syscall can return EINTR when interrupted by a signal.
#ifdef __linux__
#define WAIT_ON_FUTEX(f, v)                                                    \
  do {                                                                         \
  } while (futex((f), FUTEX_WAIT, (v), NULL, NULL, 0) == EINTR)
#endif

//--------------------------------------------------------------
// Per-thread info
//--------------------------------------------------------------

// States transitions are:
//   O|S   :: RunOwnership
//             -> WaitForSerial -> RunSerial -> WaitForOwnership
//   B|S   :: RunBuffered -> WaitForCommit -> RunCommit
//             -> WaitForSerial -> RunSerial -> WaitForBuffered
//   OB|S  :: RunBuffered -> WaitForCommit -> RunCommit
//             -> WaitForSerial -> RunSerial -> WaitForBuffered
//   O|B|S :: RunOwnership
//             -> WaitForBuffered -> RunBuffered -> WaitForCommit -> RunCommit
//             -> WaitForSerial -> RunSerial -> WaitForOwnership

enum DmpThreadState {
  Sleeping = 0, // thread is not running
  JustWokeUp,   // thread just woke up
#ifdef DMP_ENABLE_OWNERSHIP_MODE
  WaitForOwnership, // waiting for O-mode
  RunOwnership,     // running in O-mode
#endif
#ifdef DMP_ENABLE_BUFFERED_MODE
  WaitForBuffered, // waiting for B-mode
  RunBuffered,     // running in B-mode
  WaitForCommit,   // waiting for commit of B-mode
  RunCommit,       // executing commit of B-mode
#endif
  WaitForSerial,      // waiting for S-mode (possibly with fast-handoff)
  WaitForSerialToken, // waiting for S-mode token (always just after
                      // WaitForSerial)
  RunSerial,          // running in S-mode
};

struct DmpThreadInfo {
  // NOTE: keep all frequently accessed fields in the first cacheline!
  // This includes everything accessed from waitAndChange{Lock}Owner()
  // and from advanceScheduler().  We'll assume a 64B cacheline and a
  // 64bit CPU, so we have 16 ints or 8 pointers to work with.
  //
  // Note that some of these fields represent cached copies of global
  // state.  The idea is to reduce how much global state gets accessed
  // during very tight loops in the program, which hopefully reduces
  // memory bus contention.
  //
  // Current "frequently accessed" size: 64 bytes
  // (Update this comment when adding a "frequently accessed" field.)
  //
  int threadID;         // index into DMPthreadInfos[]
  DmpThreadState state; // current run state
  int schedulingChunk;  // remaining instructions in the current quantum

  // A circular queue of runnable threads.
  // This queue can only be modified when no other threads are running.
  int nextRunnableID; // copied from "nextRunnable->threadID"
  DmpThreadInfo* nextRunnable;
  DmpThreadInfo* prevRunnable;

  // Is this thread the end of the queue?
  // Used to detect when a scheduling round has ended.
  bool isLastRunnable;

  // For implementing pthread_cancel() (checked in every quantum).
  atomic_bool_t canceled;

#ifdef DMP_ENABLE_EMPTY_SERIAL_MODE
  // Do we need to execute in serial mode this round?
  atomic_bool_t needSerial;
#endif

#ifdef DMP_ENABLE_WB_HBSYNC
  // If my schedulingChunk is < this, need to update my global chunk field.
  // Checked on every commit.
  atomic_int_t triggerGlobalSchedulingChunkUpdate;
#endif

#ifdef DMP_ENABLE_EMPTY_SERIAL_MODE
  atomic_bool_t otherThreadWaitingForSerial;
  DmpThreadInfo* notifyWhenWaitingForSerial;
#endif

#ifdef DMP_ENABLE_FAST_HANDOFF
  // Spinlock to atomically test-and-wait for handoff.
  int handoffSpinlock;
  // If state == WaitForSerial, this is the resource we're blocked on.
  // If this is NULL, we're not blocked on a resource that is handoff-able.
  DMPresource* blockedOn;
#endif

#ifdef DMP_ENABLE_DATA_GROUPING
  // A stack of resources we hold.
  DMPresource* innerResource; // the innermost resource we hold, if any
  DMPresource*
      nextResource; // copied from "innerResource->outer" for fast MOT checks
#endif

#ifdef DMP_ENABLE_TINY_SERIAL_MODE
  int resourceNesting; // depth of resource nesting
#endif

  // ABOVE HERE ARE FREQUENTLY ACCESSED FIELDS.
  // NO FIELD SHOULD CROSS THIS LINE WITHOUT GOOD REASON.

  // For implementing pthread_join().
  // If another thread wants to join on us and we haven't exited.
  // they can write themselves into 'this->joiner' and sleep on
  // 'this->exited'.  When we exit, we'll wake 'this->exited'.
  atomic_int_t exited;
  DmpThreadInfo* joiner;
  pthread_t self;

  // Used by newly-woken threads.
  uint32_t roundBarrierAtStart;

  // For debugging.
  int codeLocation;

#ifdef DMP_ENABLE_MODEL_STM
  int stmThreadId;
  int stmConflicts;
  void* stmReadLog;
  void* stmWriteLog;
#endif

#ifdef DMP_ENABLE_QUANTUM_TIMING
  // Maintain TSC snapshots to measure the length of each 'DmpThreadState'.
  // At the end of each round, these are collated to produce the per-round
  // work imbalance numbers.
  struct Timing {
    uint64_t inround; // net TSC gain in this round
    double max, min;  // max/min %-of-round spent in this state
    double mean;      // mean %-of-round spent in this state
    double m2;
    // Only used on thread 0
    double spread_mean; // mean spread in units of avg-time-in-round
    double spread_m2;
  };

  Timing timing[RunSerial + 1];
  uint64_t timing_last_tsc;
  uint64_t timing_total_quanta;
#endif

#ifdef DMP_ENABLE_INSTRUMENT_WORK
  // Maintain total work for this thread, per 'DmpThreadState'.
  uint64_t work[RunSerial + 1];
  uint64_t work_this_quantum;
  uint64_t toserial_total;
  uint64_t toserial_excall;
  uint64_t toserial_mb;
  uint64_t toserial_mbresource;
  // Maintain write-buffer size stats.
#ifdef DMP_ENABLE_BUFFERED_MODE
  uint64_t wb_maxsize;
  uint64_t wb_totalsize;
  uint64_t wb_totalquanta;
  uint64_t wb_maxused;
  uint64_t wb_totalused;
  uint64_t wb_maxhashchain;
  uint64_t wb_totalhashchains;
  uint64_t wb_totalhashbuckets;
  uint64_t wb_totalcommitslocked;
  uint64_t wb_synctotal;       // HB_SYNC only
  uint64_t wb_syncwithoutwait; // HB_SYNC only
#endif                         // buffered
#endif
};

//--------------------------------------------------------------
// Global data
//--------------------------------------------------------------

// Environment configuration: constant after boot.
extern int DMP_SCHEDULING_CHUNK_SIZE;
extern int DMP_NUM_PHYSICAL_PROCESSORS;

// A list of all threads ever created (IDs are not recycled).
// NOTE: the handoff/prediction stuff assumes IDs are at most
//       16-bit, but all other code works fine with 32-bit IDs.
#define MaxThreads 1024
extern DmpThreadInfo* DMPthreadInfos[MaxThreads];
extern atomic_int_t DMPthreadInfosSize;

// The current number of live threads (always <= DMPthreadInfosSize).
extern atomic_int_t DMPnumLiveThreads;

// The runnable queue.
extern DmpThreadInfo* DMPfirstRunnable; // head
extern atomic_int_t DMPfirstRunnableID;
extern atomic_int_t DMPnumRunnableThreads; // size

// Current quantum round number
extern atomic_uint64_t DMProundNumber;

// The DMP info for the current thread, in thread-local storage.
extern __thread DmpThreadInfo* DMPMAP;

#ifdef DMP_ENABLE_WB_HBSYNC
// Global work counters.
// These are copied from DMPMAP.schedulingChunk on request from other threads
// and when parallel mode ends.  The idea is to reduce cache ping-ponging in
// DMP_waitForTurnInRound().
typedef CACHELINE_PADDED(int32_t, val) cacheline_padded_int32_t;
extern cacheline_padded_int32_t DMPglobalSchedulingChunks[MaxThreads];
#endif

//--------------------------------------------------------------
// Internal API
//--------------------------------------------------------------

void DMP_setState(DmpThreadInfo* dmp, const DmpThreadState s);
inline void DMPMAP_setState(const DmpThreadState s) { DMP_setState(DMPMAP, s); }

// library-thread.cpp
DmpThreadInfo* DMPthread_find(pthread_t thread);
void DMPthread_addToRunnableQueue(DmpThreadInfo* dmp);
void DMPthread_removeFromRunnableQueue(DmpThreadInfo* dmp);
void DMPthread_initMainThread();

// runtime-main.cpp
void DMP_printf(const char* msg, ...);

// runtime-model-*.cpp
void DMP_initRuntime();
void DMP_initRuntimeThread();
#ifdef DMP_ENABLE_BUFFERED_MODE
void DMP_commitBufferedWrites();
#endif

// runtime-sched.cpp
void DMP_waitForTurnInRound();
void DMP_waitForSerialMode();
void DMP_waitForNextQuantum();
void DMP_waitForSignal(volatile int* signal);
void DMP_sleepAndTerminate();
void DMP_resetRound(const int oldRoundBarrier);

#ifdef DMP_ENABLE_MODEL_O_B_S
void DMP_waitForBufferingMode();
#endif

#ifdef DMP_ENABLE_FAST_HANDOFF
void DMP_waitForSerialModeOrFastHandoff(DMPresource* r);
void DMP_fastHandoff(DmpThreadInfo* next, DMPresource* r);
#endif

// runtime-profiling.cpp
void DMPinstrumentation_print_thread_statistics(struct DmpThreadInfo* dmp);
void DMPprintThreadInfo(DmpThreadInfo* dmp);
void DMPprintScheduler();

#ifdef DMP_ENABLE_ROUND_TIMING
struct DmpRoundTime;
extern DmpRoundTime DMPparallelModeTime;
extern DmpRoundTime DMPcommitModeTime;
extern DmpRoundTime DMPserialModeTime;
void DMProundTimeTransition(DmpRoundTime* from, DmpRoundTime* to);
#endif

#ifdef DMP_ENABLE_QUANTUM_TIMING
extern bool DMPQuantumEndedWithAcquire;
#endif

#ifdef DMP_ENABLE_INSTRUMENT_ACQUIRES
extern void DMPinstrument_resource_acquire(DMPresource* r, int oldowner);
#endif

// Temporarily rename these so we can wrap them with instrumentation.
#if defined(DMP_ENABLE_INSTRUMENT_ACQUIRES) ||                                 \
    defined(DMP_ENABLE_INSTRUMENT_WORK) || defined(DMP_ENABLE_QUANTUM_TIMING)
#define DMP_resetRound __DMP_resetRound__
#define DMP_setState __DMP_setState__
#endif

static FORCE_INLINE uint64_t rdtsc() {
  uint32_t hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)lo) | (((uint64_t)hi) << 32);
}

//--------------------------------------------------------------
// Inline defs which must be after the above renaming
//--------------------------------------------------------------

#include "dmp-internal-resource.h"

inline void DMP_setState(DmpThreadInfo* dmp, const DmpThreadState s) {
  dmp->state = s;
}

//--------------------------------------------------------------
// For better debugging info
//--------------------------------------------------------------

#if 0 //|| defined(DMP_ENABLE_DEBUGGING)
#undef FORCE_INLINE
#define FORCE_INLINE __attribute__((noinline))
#define inline __attribute__((noinline))
#endif

#endif // _DMP_INTERNAL_H_
