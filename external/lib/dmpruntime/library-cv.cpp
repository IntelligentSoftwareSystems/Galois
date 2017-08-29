// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// PThread Library - condition variables
//

#include "dmp-internal.h"

#define CONDVAR_DEBUG_MSG(C,M,msg) DEBUG_MSG( DEBUG_CONDVAR, \
                                              msg "(%p) @%llu T:%d", \
                                              (C), DMProundNumber, DMPMAP->threadID )

struct DmpCondvarTraits {
  typedef DMPcondvar T;

  // unused:
  //   tryacquire_parallel
  //   tryacquire_serial
  //   release

  static void update_predictor(DMPresource* r, int oldowner) {
    DmpDefaultPredictor::update(r, oldowner);
  }

  static DmpThreadInfo* predict_next(DMPresource* r) {
    return DmpDefaultPredictor::predict(r);
  }

  static const bool nest_globally = false;
  static const bool acquire_ends_quantum = false;
  static const bool release_ends_quantum = false;
};

struct DmpCondvarTraitsForWait {
  typedef DMPcondvar T;

  // unused:
  //   tryacquire_parallel
  //   tryacquire_serial
  //   release

  static void update_predictor(DMPresource* r, int oldowner) {
    DmpDefaultPredictor::update(r, oldowner);
  }

  static DmpThreadInfo* predict_next(DMPresource* r) {
    return DmpDefaultPredictor::predict_ignoring(r, DMPMAP);
  }

  static const bool nest_globally = false;
  static const bool acquire_ends_quantum = false;
  static const bool release_ends_quantum = false;
};

struct CleanupData {
  DMPcondvar* C;
  DMPmutex* M;
  DMPwaiter* waiter;
};

static void cleanup(void* raw) {
  struct CleanupData* d = (struct CleanupData*)raw;
  // Remove from the wait queue.
  DMPresource_take_ownership<DmpCondvarTraits>(&d->C->resource);
  DMPwaiter_remove(&d->C->first, d->waiter);
  // Lock the given mutex.
  DMPmutex_lock(d->M);
}

//--------------------------------------------------------------
// API
//--------------------------------------------------------------

int DMPcondvar_init(DMPcondvar* C, void* attr) {
  DMP_waitForSerialMode();
  DMPresource_init(&C->resource, 0 | DMP_RESOURCE_TYPE_CONDVAR);
  C->first = NULL;
  return 0;
}

int DMPcondvar_destroy(DMPcondvar* C) {
  return 0;
}

int DMPcondvar_wait(DMPcondvar* C, DMPmutex* M) {
  CONDVAR_DEBUG_MSG(C,M,"CondvarWait-Top");
  DMPresource_take_ownership<DmpCondvarTraits>(&C->resource);

  // Add to the condvar's wait queue.
  DMPwaiter waiter = DMP_WAITER_INIT;
  DMPwaiter_add(&C->first, &waiter);

  // PThread spec says if we are canceled while waiting, then:
  // (1) We should lock 'M' before running cleanup handlers, and
  // (2) We should remove ourself from the wait queue.
  struct CleanupData d = { C, M, &waiter };
  pthread_cleanup_push(cleanup, &d);

  // Handoff ownership of the resource:
  // someone else must signal before we can make any progress!
  DMPresource_handoff<DmpCondvarTraitsForWait>(&C->resource);

  // Wait to be signaled.
  DMPmutex_unlock(M);
  DMPresource_wait_until_woken(&C->resource, &waiter);

  // Done.
  pthread_cleanup_pop(0);
  DMPmutex_lock(M);
  CONDVAR_DEBUG_MSG(C,M,"CondvarWait-Leaving");
  return 0;
}

int DMPcondvar_signal(DMPcondvar* C) {
  CONDVAR_DEBUG_MSG(C,NULL,"CondvarSignal");
  DMPresource_take_ownership<DmpCondvarTraits>(&C->resource);

  // Wake the first waiter.
  if (C->first != NULL)
    DMPwaiter_remove(&C->first, C->first);

  // Handoff so another thread can possibly wait/signal.
  DMPresource_handoff<DmpCondvarTraits>(&C->resource);
  return 0;
}

int DMPcondvar_broadcast(DMPcondvar* C) {
  CONDVAR_DEBUG_MSG(C,NULL,"CondvarBroadcast");
  DMPresource_take_ownership<DmpCondvarTraits>(&C->resource);

  // Wake all waiters.
  while (C->first != NULL)
    DMPwaiter_remove(&C->first, C->first);

  // Handoff so another thread can possibly wait/signal.
  DMPresource_handoff<DmpCondvarTraits>(&C->resource);
  return 0;
}
