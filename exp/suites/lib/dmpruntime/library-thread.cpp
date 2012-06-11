// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// PThread Library - thread management functions
//

#include "dmp-internal.h"

//-----------------------------------------------------------------------
// Utils
//-----------------------------------------------------------------------

static DmpThreadInfo* find_thread(pthread_t thread) {
  DmpThreadInfo* found = NULL;
  int id;
  for (id = 0; id < DMPthreadInfosSize; ++id) {
    if (pthread_equal(DMPthreadInfos[id]->self, thread)) {
      found = DMPthreadInfos[id];
      break;
    }
  }
  return found;
}

#ifdef DMP_ENABLE_LIBHOARD
extern void LibHoard_PthreadCreate();
extern void LibHoard_ThreadRun();
extern void LibHoard_ThreadExit();
#else
static void LibHoard_PthreadCreate() {}
static void LibHoard_ThreadRun() {}
static void LibHoard_ThreadExit() {}
#endif

//-----------------------------------------------------------------------
// Management of the runnable queue
//-----------------------------------------------------------------------

#if defined(DMP_ENABLE_MODEL_STM)
#define DMPthread_addToRunnableQueue      __DMPthread_addToRunnableQueue__
#define DMPthread_removeFromRunnableQueue __DMPthread_removeFromRunnableQueue__
#endif

void DMPthread_addToRunnableQueue(DmpThreadInfo* dmp) {
  // REQUIRES: running in serial mode!
  // Add 'dmp' to the front of the queue, assuming it's not yet on the queue.
  assert(dmp->prevRunnable == NULL);
  assert(dmp->nextRunnable == NULL);

  DmpThreadInfo* before = DMPfirstRunnable;
  DMPfirstRunnable   = dmp;
  DMPfirstRunnableID = dmp->threadID;
  ++DMPnumRunnableThreads;

  // Special case for an empty queue.
  if (before == NULL) {
    dmp->prevRunnable = dmp;
    dmp->nextRunnable = dmp;
    dmp->nextRunnableID = dmp->threadID;
    dmp->isLastRunnable = true;
    return;
  }

  // Insert 'dmp' before 'before'.
  DmpThreadInfo* prev = before->prevRunnable;
  DmpThreadInfo* next = before;
  prev->nextRunnableID = dmp->threadID;
  prev->nextRunnable = dmp;
  next->prevRunnable = dmp;
  dmp->prevRunnable = prev;
  dmp->nextRunnable = next;
  dmp->nextRunnableID = next->threadID;
}

void DMPthread_removeFromRunnableQueue(DmpThreadInfo* dmp) {
  // REQUIRES: running in serial mode!
  // Removes 'dmp' from the runnable queue, assuming it's not alone.
  assert(dmp->prevRunnable != NULL);
  assert(dmp->nextRunnable != NULL);
  assert(DMPnumRunnableThreads != 1);
  --DMPnumRunnableThreads;

  DmpThreadInfo* prev = dmp->prevRunnable;
  DmpThreadInfo* next = dmp->nextRunnable;
  prev->nextRunnableID = next->threadID;
  prev->nextRunnable = next;
  next->prevRunnable = prev;
  dmp->prevRunnable = NULL;
  dmp->nextRunnable = NULL;
  dmp->nextRunnableID = -1;

  if (DMPfirstRunnable == dmp) {
    DMPfirstRunnable   = next;
    DMPfirstRunnableID = next->threadID;
  }

  if (dmp->isLastRunnable) {
    dmp->isLastRunnable = false;
    prev->isLastRunnable = true;
  }
}

#undef DMPthread_addToRunnableQueue
#undef DMPthread_removeFromRunnableQueue

//-----------------------------------------------------------------------
// Thread destruction
//-----------------------------------------------------------------------

void DMPthread_finalize(void* raw) {
  ASSERT((DmpThreadInfo*)raw == DMPMAP);

  // Has this run yet?  This could run twice: once via our wrapper
  // around pthread_exit(), and once via our pthread cleanup hook.
  if (DMPMAP->exited)
    return;

  LibHoard_ThreadExit();

  // Finished: run DMP cleanup code.
  DMP_sleepAndTerminate();
  --DMPnumLiveThreads;
  DEBUG_MSG(DEBUG_LIFEDEATH, "Terminated thread %d, %p", DMPMAP->threadID, DMPMAP);

  // Record the completion time.
  struct timeval tv;
  gettimeofday(&tv, NULL);
  DMP_printf("DMP Thread: %p: %d: End: %ld s + %ld ms\n",
             DMPMAP, DMPMAP->threadID, tv.tv_sec, tv.tv_usec);
}

//-----------------------------------------------------------------------
// Thread creation
// Control flow:
//
//    === creator ===                  === spawnee ===
//    DMPthread_create {
//      DMP_waitForSerialMode
//      pthread_create +-------------> DMPthread_run {
//      wait until ready                 DMPthread_init
//      <-----------------------------+  signal ready
//      return                           DMP_waitForNextQuantum
//    }                                  go()
//                                     }
//-----------------------------------------------------------------------

extern void DMP_resetRound(const int oldRoundBarrier);
extern atomic_uint_t DMProundBarrier;

typedef struct DmpThreadRunArg DmpThreadRunArg;
struct DmpThreadRunArg {
  void* (*start_routine)(void*);
  void* start_arg;
  DmpThreadInfo* dmp;
  DmpThreadInfo* creator;
  volatile int ready;
};

static DmpThreadInfo* DMPthread_alloc() {
  // Allocate a thread info and add it to the global list of threads.
  DmpThreadInfo* dmp = (DmpThreadInfo*)alloc_cache_aligned(sizeof *dmp);

  const int id = DMPthreadInfosSize;
  if (id >= MaxThreads) {
    ERROR_MSG("out of threads\n");
    exit(99);
  }

  ASSERT(DMPthreadInfos[id] == NULL);
  DMPthreadInfos[id] = dmp;
  ++DMPthreadInfosSize;
  ++DMPnumLiveThreads;

  // Do basic initialization on 'dmp': we finish in DMPthread_init().
  dmp->threadID = id;
  dmp->exited = 0;

  return dmp;
}

static void DMPthread_init(DmpThreadInfo* dmp) {
  LibHoard_ThreadRun();

  // Finish initializing the DmpThreadInfo.
  // NOTE: we are running in serial mode on behalf of 'creator'.
  DMPMAP = dmp;
  DMPMAP->self = pthread_self();
  DMPMAP->state = JustWokeUp;
  DMPMAP->schedulingChunk = DMP_SCHEDULING_CHUNK_SIZE;
  DMPMAP->roundBarrierAtStart = DMProundBarrier;

  DMPthread_addToRunnableQueue(DMPMAP);
  DMP_initRuntimeThread();

  const int id = DMPMAP->threadID;
  DEBUG_MSG(DEBUG_LIFEDEATH, "DMPthread_init: a thread %d is born\n", id);
  DEBUG_MSG(DEBUG_LIFEDEATH, "Live threads: %d", DMPnumLiveThreads);
}

static void* DMPthread_run(void *raw) {
  // This new thread starts running on behalf of 'creator', who is
  // blocked waiting for us to finish initialization, so finish and
  // then notify our creator that we're ready to run.
  DmpThreadRunArg* run = (DmpThreadRunArg*)raw;
  DMPthread_init(run->dmp);
  run->ready = 1;
  __sync_synchronize();

  // Thread 0 is the one that runs static destructors.
  // It should NOT ever be created by this function!
  ASSERT(DMPMAP->threadID != 0);

  struct timeval tv;
  gettimeofday(&tv, NULL);
  DMP_printf("DMP Thread: %d: Start: %ld s + %ld ms\n",
             DMPMAP->threadID, tv.tv_sec, tv.tv_usec);

  // Go! Setup cleanup a handler in case someone calls pthread_exit().
  void* r;
  pthread_cleanup_push(DMPthread_finalize, DMPMAP);
  DMP_waitForNextQuantum();
  r = run->start_routine(run->start_arg);
  pthread_cleanup_pop(1 /* execute */);
  return r;
}

int DMPthread_create(pthread_t *thread, pthread_attr_t *attr,
                     void *(*start_routine)(void*), void *start_arg) {
  pthread_attr_t the_attr;
  pthread_t the_thread;

  DEBUG_MSG(DEBUG_LIFEDEATH, "In DMPthread_create()");
  LibHoard_PthreadCreate();

  if (!thread) {
    thread = &the_thread;
  }
  if (!attr) {
    pthread_attr_init(&the_attr);
    attr = &the_attr;
  }

  DMP_waitForSerialMode();
  DmpThreadInfo* dmp = DMPthread_alloc();

  const int id = dmp->threadID;
  DEBUG_MSG(DEBUG_LIFEDEATH, "DMPthread_create: creating thread %d\n", id);

  // Now create the thread and block until it is ready to run.
  DmpThreadRunArg* run = (DmpThreadRunArg*)alloc_cache_aligned(sizeof *run);
  run->start_routine = start_routine;
  run->start_arg = start_arg;
  run->dmp = dmp;
  run->creator = DMPMAP;
  run->ready = 0;
  const int r = pthread_create(thread, attr, DMPthread_run, (void*)run);

  if (r == 0) {
    while (!__sync_bool_compare_and_swap(&run->ready, 1, 1))
      YIELD();
    DEBUG_MSG(DEBUG_LIFEDEATH, "DMPthread_create() succeeded: id=%d", id);
  } else {
    DEBUG_MSG(DEBUG_LIFEDEATH, "DMPthread_create() failed: error=%d", r);
  }

  return r;
}

void DMPthread_initMainThread() {
  // Setup a thread for main() to invoke.
  // This is a trimmed-down version of DMPthread_create().

  // Allocate the first thread info.
  ASSERT(DMPthreadInfosSize == 0);
  ASSERT(DMPthreadInfos[0] == NULL);

  DmpThreadInfo* dmp = DMPthread_alloc();
  DMPthread_init(dmp);

  // Start the first quantum.
  DMPMAP->roundBarrierAtStart = 1 - DMProundBarrier;
  DMP_waitForNextQuantum();
  DMP_resetRound(DMProundBarrier);
}

//-----------------------------------------------------------------------
// Thread join and cancelation
//-----------------------------------------------------------------------

int DMPthread_join(pthread_t thread, void **value_ptr) {
  // This doesn't need to wait until serial mode. Consider this
  // analogy: 'exited' is a shared variable which written by the
  // exiting thread and read by the joiner thread.  Obviously,
  // the exiting thread must write to 'exited' in serial mode.

  DmpThreadInfo* dmp = find_thread(thread);
  if (dmp == NULL) return ESRCH;
  if (dmp == DMPMAP) return EDEADLK;
  if (dmp->joiner != NULL) return EINVAL;

  // Wait for the thread to die.
  if (!dmp->exited) {
    DMP_waitForSerialMode();
    if (!dmp->exited) {
      if (dmp->joiner != NULL)
        return EINVAL;
      dmp->joiner = DMPMAP;
      DMP_waitForSignal((volatile int*)&(dmp->exited));
    }
  }

  // Now join on the thread.
  int ret = pthread_join(thread, value_ptr);
  dmp->self = 0x0;
  return ret;
}

int DMPthread_cancel(pthread_t thread) {
  DMP_waitForSerialMode();

  DmpThreadInfo* dmp = find_thread(thread);
  if (dmp == NULL) return ESRCH;

  // Mark the thread canceled.
  dmp->canceled = true;

  // Return success: the thread will cancel itself
  // the next time it ends a quantum.
  return 0;
}
