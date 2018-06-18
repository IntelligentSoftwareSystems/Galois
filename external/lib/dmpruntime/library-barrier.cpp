// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// PThread Library - barriers
//

#include "dmp-internal.h"

#define BARRIER_DEBUG_MSG(B, msg)                                              \
  DEBUG_MSG(DEBUG_BARRIER, msg "(%p) @%llu T:%d arrived:%d", (B),              \
            DMProundNumber, DMPMAP->threadID, (B)->arrived)

struct DmpBarrierPredictor {
  // Since no thread blocked at a barrier can make progress until
  // other threads arrive at the barrier, always handoff to some
  // thread not blocked at the barrier.

  static void update(DMPresource* r, int oldowner) {}

  static DmpThreadInfo* predict(DMPresource* r) {
    DMPbarrier* B = RESOURCE_CONTAINER(r, DMPbarrier);

    if (B->first == NULL)
      return NULL;

    DmpThreadInfo* dmp;
    for (dmp = DMPMAP->nextRunnable; dmp != DMPMAP; dmp = dmp->nextRunnable) {
      bool found = false;
      for (DMPwaiter* w = B->first; w; w = w->next) {
        if (w->dmp == dmp) {
          found = true;
          break;
        }
      }
      if (!found)
        return dmp;
    }

    return NULL;
  }
};

struct DmpBarrierTraits {
  typedef DMPbarrier T;

  // unused:
  //   tryacquire_parallel
  //   tryacquire_serial
  //   release

#ifdef DMP_ENABLE_PREDICT_HANDOFF_BARRIER
  typedef DmpBarrierPredictor Predictor;
#else
  typedef DmpDefaultPredictor Predictor;
#endif

  static void update_predictor(DMPresource* r, int oldowner) {
    Predictor::update(r, oldowner);
  }
  static DmpThreadInfo* predict_next(DMPresource* r) {
    return Predictor::predict(r);
  }

  static const bool nest_globally        = false;
  static const bool acquire_ends_quantum = false;
  static const bool release_ends_quantum = false;
};

//--------------------------------------------------------------
// API
//--------------------------------------------------------------

//
// PThread-style barriers
// 'needed' is specified on init
//

#ifndef PTHREAD_BARRIER_SERIAL_THREAD
#define PTHREAD_BARRIER_SERIAL_THREAD -1
#endif

int DMPbarrier_init(DMPbarrier* B, void* attr, unsigned needed) {
  if (needed == 0)
    return EINVAL;
  DMP_waitForSerialMode();
  DMPresource_init(&B->resource, 0 | DMP_RESOURCE_TYPE_BARRIER);
  B->needed  = needed;
  B->arrived = 0;
  B->first   = NULL;
  return 0;
}

int DMPbarrier_destroy(DMPbarrier* B) {
  DMPresource_take_ownership<DmpBarrierTraits>(&B->resource);
  if (B->arrived > 0)
    return EBUSY;
  B->needed  = 0;
  B->arrived = 0;
  return 0;
}

int DMPbarrier_wait(DMPbarrier* B) {
  return DMPbarrier_wait_splash(B, B->needed);
}

//
// Splash-style barriers
// 'needed' is specified on wait
//

int DMPbarrier_init_splash(DMPbarrier* B) {
  DMP_waitForSerialMode();
  DMPresource_init(&B->resource, 0 | DMP_RESOURCE_TYPE_BARRIER);
  B->needed  = 0; // unused
  B->arrived = 0;
  B->first   = NULL;
  return 0;
}

int DMPbarrier_wait_splash(DMPbarrier* B, unsigned needed) {
  BARRIER_DEBUG_MSG(B, "BarrierArrive");
  DMPresource_take_ownership<DmpBarrierTraits>(&B->resource);
  B->arrived++;

  int r;
  if (B->arrived >= needed) {
    BARRIER_DEBUG_MSG(B, "BarrierOpen");
    // Wake all other threads.
    while (B->first != NULL)
      DMPwaiter_remove(&B->first, B->first);
    // Reset.
    B->arrived = 0;
    r          = PTHREAD_BARRIER_SERIAL_THREAD;
  } else {
    BARRIER_DEBUG_MSG(B, "BarrierWait");
    // Wait to be woken.
    DMPwaiter waiter = DMP_WAITER_INIT;
    DMPwaiter_add(&B->first, &waiter);
    DMPresource_handoff<DmpBarrierTraits>(&B->resource);
    DMPresource_wait_until_woken(&B->resource, &waiter);
    r = 0;
  }

  BARRIER_DEBUG_MSG(B, "BarrierLeave");
  return r;
}
