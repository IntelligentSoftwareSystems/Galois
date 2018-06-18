// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// PThread Library - semaphores
//

#include "dmp-internal.h"

#define SEM_DEBUG_MSG(S, msg)                                                  \
  DEBUG_MSG(DEBUG_MUTEX, msg "(%p) @%llu T:%d", (S), DMProundNumber,           \
            DMPMAP->threadID)

inline bool sem_tryacquire(DMPsemaphore* sem, DMPwaiter* w) {
  if (sem->count) {
    sem->count--;
    __sync_synchronize();
    return true;
  } else {
    return false;
  }
}

inline void sem_release(DMPsemaphore* sem) {
  sem->count++;
  __sync_synchronize();
}

struct DmpSemaphoreTraits {
  typedef DMPsemaphore T;
  typedef DmpResourceTryacquireWrapper<DMPsemaphore, sem_tryacquire> Wrapper;

  static bool tryacquire_parallel(DMPsemaphore* sem, DMPwaiter* w) {
    return Wrapper::tryacquire_parallel(sem, w);
  }

  static bool tryacquire_serial(DMPsemaphore* sem, DMPwaiter* w) {
    return Wrapper::tryacquire_serial(sem, w);
  }

  static void release(DMPsemaphore* sem) { sem_release(sem); }

  static void update_predictor(DMPresource* r, int oldowner) {
    DmpDefaultPredictor::update(r, oldowner);
  }

  static DmpThreadInfo* predict_next(DMPresource* r) {
    return DmpDefaultPredictor::predict(r);
  }

  static const bool nest_globally        = false;
  static const bool acquire_ends_quantum = false;
  static const bool release_ends_quantum = false;
};

//--------------------------------------------------------------
// API
//--------------------------------------------------------------

int DMPsemaphore_init(DMPsemaphore* sem, int pshared, unsigned value) {
  DMP_waitForSerialMode();
  DMPresource_init(&sem->resource, 0 | DMP_RESOURCE_TYPE_SEM);
  sem->count = value;
  return 0;
}

int DMPsemaphore_post(DMPsemaphore* sem) {
  SEM_DEBUG_MSG(sem, "SemPost");
#ifdef DMP_ENABLE_HANDOFF
  DMPresource_release<DmpSemaphoreTraits>(&sem->resource, sem, true);
#else
  DMPresource_release<DmpSemaphoreTraits>(&sem->resource, sem, false);
#endif
  return 0;
}

int DMPsemaphore_wait(DMPsemaphore* sem) {
  SEM_DEBUG_MSG(sem, "SemWait");
  DMPresource_acquire<DmpSemaphoreTraits>(&sem->resource, sem);
  return 0;
}
