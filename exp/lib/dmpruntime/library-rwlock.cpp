// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// PThread Library - reader/writer locks
//
// This implementation gives preference to writers.
// We assume that the read frequency is much higher than the
// write frequency (this is a normal assumption).  POSIX does
// not guarantee a fair lock, so we won't either.
//

#include "dmp-internal.h"

#define RWLOCK_DEBUG_MSG(L,msg) DEBUG_MSG( DEBUG_MUTEX, \
                                           msg "(%p) @%llu T:%d", \
                                           (L), DMProundNumber, DMPMAP->threadID )

// For lock(), this is how we adjust waiting_writers:
//   w->rounds == 0 :: increment
//   w->rounds  > 0 :: don't change (already incremented)
//   w == NULL      :: don't change (calling trylock, not lock)
//
// TODO: possible resource enhancement:
//   Separate queues for writers and readers

inline bool rwlock_rd_tryacquire(DMPrwlock* rwlock, DMPwaiter* w) {
  if (rwlock->state >= 0 && rwlock->waiting_writers == 0) {
    rwlock->state++;
    __sync_synchronize();
    return true;
  } else {
    return false;
  }
}

inline bool rwlock_wr_tryacquire(DMPrwlock* rwlock, DMPwaiter* w) {
  if (rwlock->state == 0) {
    rwlock->state = -1;
    if (w && w->rounds > 0) rwlock->waiting_writers--;
    __sync_synchronize();
    return true;
  } else {
    if (w && w->rounds == 0) rwlock->waiting_writers++;
    return false;
  }
}

inline void rwlock_release(DMPrwlock* rwlock) {
  if (rwlock->state <= 0) {
    rwlock->state = 0;
  } else {
    rwlock->state--;
  }
  __sync_synchronize();
}

struct DmpRdlockTraits {
  typedef DMPrwlock T;
  typedef DmpResourceTryacquireWrapper<DMPrwlock, rwlock_rd_tryacquire> Wrapper;

  static bool tryacquire_parallel(DMPrwlock* rwlock, DMPwaiter* w) {
    return Wrapper::tryacquire_parallel(rwlock, w);
  }

  static bool tryacquire_serial(DMPrwlock* rwlock, DMPwaiter* w) {
    return Wrapper::tryacquire_serial(rwlock, w);
  }

  static void release(DMPrwlock* rwlock) {
    rwlock_release(rwlock);
  }

  static void update_predictor(DMPresource* r, int oldowner) {
    DmpDefaultPredictor::update(r, oldowner);
  }

  static DmpThreadInfo* predict_next(DMPresource* r) {
    return DmpDefaultPredictor::predict(r);
  }

#if defined(DMP_ENABLE_DATA_GROUP_BY_MUTEX) || defined(DMP_ENABLE_TINY_SERIAL_MODE)
  static const bool nest_globally = true;
#else
  static const bool nest_globally = false;
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

struct DmpWrlockTraits {
  typedef DMPrwlock T;
  typedef DmpResourceTryacquireWrapper<DMPrwlock, rwlock_wr_tryacquire> Wrapper;

  static bool tryacquire_parallel(DMPrwlock* rwlock, DMPwaiter* w) {
    return Wrapper::tryacquire_parallel(rwlock, w);
  }

  static bool tryacquire_serial(DMPrwlock* rwlock, DMPwaiter* w) {
    return Wrapper::tryacquire_serial(rwlock, w);
  }

  // Same as DmpRdlockTraits

  static void release(DMPrwlock* rwlock) {
    rwlock_release(rwlock);
  }
  static void update_predictor(DMPresource* r, int oldowner) {
    DmpDefaultPredictor::update(r, oldowner);
  }
  static DmpThreadInfo* predict_next(DMPresource* r) {
    return DmpDefaultPredictor::predict(r);
  }

  static const bool nest_globally = DmpRdlockTraits::nest_globally;
  static const bool acquire_ends_quantum = DmpRdlockTraits::acquire_ends_quantum;
  static const bool release_ends_quantum = DmpRdlockTraits::release_ends_quantum;
};

//--------------------------------------------------------------
// API
//--------------------------------------------------------------

int DMPrwlock_init(DMPrwlock* rwlock, void* attr) {
  ASSERT(attr == NULL);
  DMP_waitForSerialMode();
  DMPresource_init(&rwlock->resource, 0 | DMP_RESOURCE_TYPE_RWLOCK);
  rwlock->state = 0;
  rwlock->waiting_writers = 0;
  return 0;
}

int DMPrwlock_destroy(DMPrwlock* rwlock) {
  DMPresource_take_ownership<DmpRdlockTraits>(&rwlock->resource);
  if (rwlock->state != 0)
    return EBUSY;
  else
    return 0;
}

int DMPrwlock_rdlock(DMPrwlock* rwlock) {
  RWLOCK_DEBUG_MSG(rwlock, "ReadLock");
  DMPresource_acquire<DmpRdlockTraits>(&rwlock->resource, rwlock);
  return 0;
}

int DMPrwlock_tryrdlock(DMPrwlock* rwlock) {
  if (DMPresource_tryacquire<DmpRdlockTraits>(&rwlock->resource, rwlock, NULL))
    return 0;
  else
    return EBUSY;
}

int DMPrwlock_wrlock(DMPrwlock* rwlock) {
  RWLOCK_DEBUG_MSG(rwlock, "WriteLock");
  DMPresource_acquire<DmpWrlockTraits>(&rwlock->resource, rwlock);
  return 0;
}

int DMPrwlock_trywrlock(DMPrwlock* rwlock) {
  if (DMPresource_tryacquire<DmpWrlockTraits>(&rwlock->resource, rwlock, NULL))
    return 0;
  else
    return EBUSY;
}

int DMPrwlock_unlock(DMPrwlock* rwlock) {
  RWLOCK_DEBUG_MSG(rwlock, "RwUnlock");
#ifdef DMP_ENABLE_HANDOFF
  DMPresource_release<DmpRdlockTraits>(&rwlock->resource, rwlock, true);
#else
  DMPresource_release<DmpRdlockTraits>(&rwlock->resource, rwlock, false);
#endif
  return 0;
}
