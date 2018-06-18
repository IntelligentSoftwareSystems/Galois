// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// PThread Library - mutexes
//

#include "dmp-internal.h"

#define MUTEX_DEBUG_MSG(M, msg)                                                \
  DEBUG_MSG(DEBUG_MUTEX, msg "(%p) @%llu T:%d", (M), DMProundNumber,           \
            DMPMAP->threadID)

inline bool mutex_tryacquire(DMPmutex* mutex, DMPwaiter* w) {
  return DMP_SPINLOCK_TRYLOCK(&mutex->spinlock);
}

inline bool mutex_isheld(DMPmutex* mutex) { return mutex->spinlock != 0; }

inline void mutex_release(DMPmutex* mutex) {
  DMP_SPINLOCK_UNLOCK(&mutex->spinlock);
}

struct DmpMutexTraits {
  typedef DMPmutex T;
  typedef DmpResourceTryacquireWrapper<DMPmutex, mutex_tryacquire> Wrapper;

  static bool tryacquire_parallel(DMPmutex* mutex, DMPwaiter* w) {
    return Wrapper::tryacquire_parallel(mutex, w);
  }

  static bool tryacquire_serial(DMPmutex* mutex, DMPwaiter* w) {
    return Wrapper::tryacquire_serial(mutex, w);
  }

  static void release(DMPmutex* mutex) { mutex_release(mutex); }

  static void update_predictor(DMPresource* r, int oldowner) {
    DmpDefaultPredictor::update(r, oldowner);
  }

  static DmpThreadInfo* predict_next(DMPresource* r) {
    return DmpDefaultPredictor::predict(r);
  }

#if defined(DMP_ENABLE_DATA_GROUP_BY_MUTEX) ||                                 \
    defined(DMP_ENABLE_TINY_SERIAL_MODE)
  static const bool nest_globally = true;
#else
  static const bool nest_globally        = false;
#endif

#ifdef DMP_ENABLE_MUTEX_LOCK_ENDQUANTUM
  static const bool acquire_ends_quantum = true;
#else
  static const bool acquire_ends_quantum = false;
#endif

#ifdef DMP_ENABLE_MUTEX_UNLOCK_ENDQUANTUM
  static const bool release_ends_quantum = true;
#else
  static const bool release_ends_quantum = false;
#endif
};

//--------------------------------------------------------------
// API
//--------------------------------------------------------------

int DMPmutex_init(DMPmutex* mutex, void* attr) {
  DMP_waitForSerialMode();
  mutex->spinlock = 0;
  DMPresource_init(&mutex->resource, 0 | DMP_RESOURCE_TYPE_MUTEX);
  return 0;
}

int DMPmutex_destroy(DMPmutex* mutex) {
  DMPresource_take_ownership<DmpMutexTraits>(&mutex->resource);
  if (mutex->spinlock)
    return EBUSY;
  else
    return 0;
}

int DMPmutex_lock(DMPmutex* mutex) {
  MUTEX_DEBUG_MSG(mutex, "MutexLock");
  DMPresource_acquire<DmpMutexTraits>(&mutex->resource, mutex);
  return 0;
}

int DMPmutex_trylock(DMPmutex* mutex) {
  if (DMPresource_tryacquire<DmpMutexTraits>(&mutex->resource, mutex, NULL))
    return 0;
  else
    return EBUSY;
}

int DMPmutex_unlock(DMPmutex* mutex) {
  MUTEX_DEBUG_MSG(mutex, "MutexUnlock");
  // See comments for DMPresource_release().
#ifdef DMP_ENABLE_WB_HBSYNC_UNLOCKOPT
  const bool needs_ownership =
      (!mutex_isheld(mutex) || !DMPresource_is_owner(&mutex->resource, DMPMAP));
#elif defined(DMP_ENABLE_HANDOFF)
  const bool needs_ownership             = true;
#else
  const bool needs_ownership = false;
#endif
  DMPresource_release<DmpMutexTraits>(&mutex->resource, mutex, needs_ownership);
  return 0;
}
