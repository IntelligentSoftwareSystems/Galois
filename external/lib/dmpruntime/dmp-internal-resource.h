// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Templated interface to the resource library.
// Include only from dmp-internal.h
//

#ifndef _DMP_INTERNAL_RESOURCE_H_
#define _DMP_INTERNAL_RESOURCE_H_

#define RESOURCE_CONTAINER(ptr, type)                                          \
  ((type*)((char*)(ptr)-offsetof(type, resource)))

//--------------------------------------------------------------
// Resource basic API
//--------------------------------------------------------------

inline void DMPresource_init(DMPresource* r, int state) {
  dmp_static_assert(DMP_RESOURCE_STATE_OWNER_MASK >= MaxThreads);
  r->state   = state;
  r->waiters = NULL;
#ifdef DMP_ENABLE_WB_HBSYNC
  r->lastRoundUsed = 0;
#endif
#ifdef DMP_ENABLE_DATA_GROUPING
  r->outer = NULL;
#endif
#ifdef DMP_ENABLE_PREDICT_HANDOFF_WINDOWED
  for (int i = 0; i < ARRAY_SIZE(r->recentAcquires); ++i)
    r->recentAcquires[i] = -1;
  r->recentAcquiresSlot = 0;
#endif
#ifdef DMP_ENABLE_PREDICT_HANDOFF_MARKOV
  for (int i = 0; i < ARRAY_SIZE(r->acquires); ++i) {
    r->acquires[i].threadID            = -1;
    r->acquires[i].recentFollowersSlot = 0;
    for (int k = 0; k < ARRAY_SIZE(r->acquires[i].recentFollowers); ++k)
      r->acquires[i].recentFollowers[k] = -1;
  }
#endif
#ifdef DMP_ENABLE_INSTRUMENT_ACQUIRES
  r->lastRoundAcquired = 0;
#endif
}

inline int DMPresource_owner(DMPresource* r) {
  return r->state & DMP_RESOURCE_STATE_OWNER_MASK;
}

inline bool DMPresource_is_owner(DMPresource* r, DmpThreadInfo* dmp) {
  return DMPresource_owner(r) == dmp->threadID;
}

inline void DMPresource_set_owner(DMPresource* r, DmpThreadInfo* dmp) {
  r->state = dmp->threadID | (r->state & ~DMP_RESOURCE_STATE_OWNER_MASK);
#ifdef DMP_ENABLE_WB_HBSYNC
  r->lastRoundUsed = DMProundNumber; // mark this resource used
#endif
}

inline int DMPresource_type(DMPresource* r) {
  return r->state & DMP_RESOURCE_STATE_TYPE_MASK;
}

//--------------------------------------------------------------
// Wait queues
//--------------------------------------------------------------

static void DMPwaiter_add(DMPwaiter** first, DMPwaiter* w) {
  // REQUIRES: 'w' is not attached to any list
  DMP_ASSERT(w);
  w->waiting = 1;
  w->dmp     = DMPMAP;
  w->next    = NULL;
#ifdef DMP_ENABLE_WB_HBSYNC
  w->roundReleased = (uint64_t)(-1);
#endif
  while (*first)
    first = &(*first)->next;
  *first = w;
}

static void DMPwaiter_remove(DMPwaiter** first, DMPwaiter* w) {
  // REQUIRES: 'w' is attached to '*first'
  DMP_ASSERT(w);
  while (*first && *first != w) {
    first = &(*first)->next;
  }
  if (*first == w) {
    *first = w->next;
  }
#ifdef DMP_ENABLE_WB_HBSYNC
  w->roundReleased = DMProundNumber;
#endif
  w->waiting = 0;
  w->next    = NULL;
}

//--------------------------------------------------------------
// Memory consistency
//--------------------------------------------------------------

static bool DMPresource_ownership_barrier(DMPresource* const r) {
  // Returns 'true' iff the resource can be acquired without a round barrier.
#if defined(DMP_ENABLE_WB_THREADLOCAL_SYNCOPS)
  {
    // Check MOT
    if (DMPresource_owner(r) != DMPMAP->threadID)
      DMPmembarrierResource();
    return true;
  }
#elif defined(DMP_ENABLE_WB_HBSYNC)
  {
    DMP_ASSERT(r->lastRoundUsed <= DMProundNumber);

    // A resource can be owned by just one thread per quantum.
    if (r->lastRoundUsed == DMProundNumber)
      return DMPresource_is_owner(r, DMPMAP);

    // No one has taken ownership for this quantum yet: try again.
    DMP_waitForTurnInRound();
    return (r->lastRoundUsed < DMProundNumber);
  }
#else
  {
    // For B|S, this will enter serial mode.
    // For O|S, this is a nop.
    DMPmembarrierResource();
    return true;
  }
#endif
}

//--------------------------------------------------------------
// Resource Trails
//
// This struct is provided as the template argument to all
// operations, and it should have the following form:
//
//   struct Traits {
//     // Resource type (e.g., DMPmutex or DMPbarrier).
//     typedef T;
//
//     static bool tryacquire_parallel(T*, DMPwaiter*);
//     static bool tryacquire_serial(T*, DMPwaiter*);
//     static void release(T*);
//
//     // Handoff prediction.
//     static void update_predictor(DMPresource*, int oldowner);
//     static int  predict_next(DMPresource*);
//
//     // If true, the resource should be included in the per-thread
//     // acquired resource stack ('DMPMAP->innerResource').
//     static const bool nest_globally;
//
//     // Quantum building: used only for mutexes and rwlocks.
//     static const bool acquire_ends_quantum;
//     static const bool release_ends_quantum;
//   }
//--------------------------------------------------------------

//--------------------------------------------------------------
// TryAcquire / Handoff
// These are the primitive operations.
//
// DMPresource_tryacquire()
//   * Succeeds iff a resource can be acquired, right now, where
//     "can be acquired" is defined by two functions:
//       -- Traits::tryacquire_parallel
//       -- Traits::tryacquire_serial
//     The first is called if the thread might be running in parallel
//     mode without ownership of the resource.  The second is called
//     when the thread is running in serial mode, or in parallel mode
//     and definately owns the resource.
//   * Supports slow-handoff and fast-handoff.
//   * OPTIONALLY:
//       -- updates resource nesting ('DMPMAP->innerResource')
//       -- updates handoff predictors ('Traits::update_predictor')
//
// DMPresource_handoff()
//   * Hands ownership to the next thread, where the "next" thread is:
//       1. the first thread in 'waiters', if any
//       2. the next predicted thread ('Traits::predict_next')
//   * REQUIRES:
//       -- the current thread owns the resource
//   * OPTIONALLY:
//       -- updates resource nesting ('DMPMAP->innerResource')
//--------------------------------------------------------------

template <typename Traits>
bool DMPresource_tryacquire(DMPresource* const r,
                            typename Traits::T* const object,
                            DMPwaiter* const waiter) {
  const bool cancontinue = DMPresource_ownership_barrier(r);

#ifdef DMP_ENABLE_INSTRUMENT_WORK
  DMPMAP->wb_synctotal++;
#ifdef DMP_ENABLE_WB_HBSYNC
  if (cancontinue)
    DMPMAP->wb_syncwithoutwait++;
#endif
#endif

  if (!cancontinue)
    return false;

  const int oldowner = DMPresource_owner(r);

#ifdef DMP_ENABLE_MODEL_STM
  extern void DMPstoreContained(void* addr);
  DMPstoreContained(r);
#endif

  // In serial mode, we can immediately try to acquire.
  // We won't steal ownership unless we successfuly acquire.
  if (DMPMAP->state == RunSerial) {
    if (Traits::tryacquire_serial(object, waiter))
      goto success;
    else
      return false;
  }

#if !defined(DMP_ENABLE_FAST_HANDOFF)
  {
    // In parallel mode: try to acquire now, if we own the resource.
    if (Traits::tryacquire_parallel(object, waiter))
      goto success;

    // Try again in serial mode.
    DMP_waitForSerialMode();
    if (Traits::tryacquire_serial(object, waiter))
      goto success;
  }
#else // DMP_ENABLE_FAST_HANDOFF
  {
    // In parallel mode: try to acquire if we own the resource.
    // See DMP_fastHandoff() for the fast-handoff protocol.
    if (Traits::tryacquire_parallel(object, waiter))
      goto success;

    // Double-checked.
    DMP_SPINLOCK_LOCK(&DMPMAP->handoffSpinlock);

    if (Traits::tryacquire_parallel(object, waiter)) {
      DMP_SPINLOCK_UNLOCK(&DMPMAP->handoffSpinlock);
      goto success;
    }

    // Wait until serial mode or we're passed the resource, whichever is first.
    // This will unlock the handoff spinlock.
    DMP_waitForSerialModeOrFastHandoff(r);

    // If we're back in parallel mode, we MUST have ownership.
    if (Traits::tryacquire_serial(object, waiter))
      goto success;
  }
#endif

  return false;

success:
#ifdef DMP_ENABLE_INSTRUMENT_ACQUIRES
  DMPinstrument_resource_acquire(r, oldowner);
#endif

#ifdef DMP_ENABLE_DATA_GROUPING
  // Add to the held-resource chain.
  if (Traits::nest_globally) {
    r->outer              = DMPMAP->innerResource;
    DMPMAP->innerResource = r;
    DMPMAP->nextResource  = r->outer;
  }
#endif

#ifdef DMP_ENABLE_TINY_SERIAL_MODE
  // Update resource nesting level.
  if (Traits::nest_globally) {
    DMPMAP->resourceNesting++;
  }
#endif

#ifdef DMP_ENABLE_WB_HBSYNC
  // On success we should have ownership of this resource.
  DMP_ASSERT(r->lastRoundUsed == DMProundNumber);
  DMP_ASSERT(DMPresource_owner(r) == DMPMAP->threadID);
#endif

  // Handoff predictors.
  Traits::update_predictor(r, oldowner);

  // Success ends the quantum?
  if (Traits::acquire_ends_quantum) {
    if ((r->state & DMP_RESOURCE_STATE_USED) != 0) {
      DMP_waitForNextQuantum();
    } else {
      r->state |= DMP_RESOURCE_STATE_USED;
    }
  }

  return true;
}

template <typename Traits>
void DMPresource_handoff(DMPresource* const r) {
#if defined(DMP_ENABLE_DATA_GROUPING) || defined(DMP_ENABLE_HANDOFF)
  DMP_ASSERT(DMPresource_owner(r) == DMPMAP->threadID);
#endif

#ifdef DMP_ENABLE_DATA_GROUPING
  // Remove from the held-resource chain.
  // NOTE: The chain will be broken if some thread T releases a resource
  // that was acquired by some other thread T' (can happen for mutexes).
  if (Traits::nest_globally) {
    if (DMPMAP->innerResource == r) {
      DMPMAP->innerResource = r->outer;
      DMPMAP->nextResource  = (r->outer) ? r->outer->outer : NULL;
      r->outer              = NULL;
    } else {
      DMPresource *p, *n;
      for (n = DMPMAP->innerResource, n = NULL; n; p = n, n = n->outer) {
        if (n == r) {
          if (p)
            p->outer = n->outer;
          if (DMPMAP->nextResource == n)
            DMPMAP->nextResource = n->outer;
          r->outer = NULL;
          break;
        }
      }
    }
  }
#endif

  // Handoff: slow-handoff can only happen in serial mode.
#ifdef DMP_ENABLE_HANDOFF
#ifndef DMP_ENABLE_FAST_HANDOFF
  if (DMPMAP->state == RunSerial)
#endif
  {
    DmpThreadInfo* next = NULL;

    // Handoff to a blocked thread, if any.
    if (r->waiters != NULL) {
      next = r->waiters->dmp;
    }

    // Handoff to a predicted thread, if any.
    if (next == NULL) {
      next = Traits::predict_next(r);
    }

    // Perform the handoff.
    if (next != NULL && DMPresource_owner(r) != next->threadID) {
      DMPresource_set_owner(r, next);
#ifdef DMP_ENABLE_FAST_HANDOFF
      DMP_fastHandoff(next, r);
#endif
    }
  }
#endif
}

//--------------------------------------------------------------
// High Level Operations
// These build from the primitive operations.
//
// DMPresource_acquire()
//   * Blocks until 'tryacquire' succeeds.
//   * While blocked, the thread is added to 'r->waiters'.
//
// DMPresource_take_ownership()
//   * Blocks until the thread can take ownership of the resource,
//     then takes ownership and returns.
//
// DMPresource_release()
//   * Equivalent to:
//       1. DMPresource_take_ownership()
//       2. Traits::release()
//       3. DMPresource_handoff()
//
// DMPresource_wait_until_woken()
//   * Blocks until the thread is removed from a wait queue, given
//     an entry that must be attached to some wait queue associated
//     with the given resource (not necessarily 'r->waiters').
//   * This does not updated handoff predictors.
//   * This is used to implement DMPbarrier_wait and DMPcondvar_wait.
//--------------------------------------------------------------

template <typename Traits>
void DMPresource_acquire(DMPresource* const r,
                         typename Traits::T* const object) {
  DMPwaiter waiter = DMP_WAITER_INIT;
  waiter.dmp       = DMPMAP;

  while (true) {
    if (DMPresource_tryacquire<Traits>(r, object, &waiter)) {
      if (waiter.waiting)
        DMPwaiter_remove(&r->waiters, &waiter);
      return;
    }

#ifdef DMP_ENABLE_HANDOFF
    // Right now we're running in serial mode.  If not yet waiting,
    // add ourselves to the wait queue.  Note that this DOES NOT
    // guarantee FIFO acquire order of mutexes, since another thread
    // can acquire in serial mode even if we're first on 'r->waiters'.
    // TODO: pull this thread off the runnable queue entirely?
    waiter.rounds++;
    if (!waiter.waiting) {
      DMPwaiter_add(&r->waiters, &waiter);
    }
#endif

#ifdef DMP_ENABLE_QUANTUM_TIMING
    DMPQuantumEndedWithAcquire = true;
#endif
    DMP_waitForNextQuantum();
  }
}

template <typename TraitsBase>
struct DmpResourceTakeOwnershipTraits {
  typedef typename TraitsBase::T T;
  static bool tryacquire_parallel(T* object, DMPwaiter*) {
#ifdef DMP_ENABLE_WB_HBSYNC
    // In this case we've guaranteed that it's our deterministic turn.
    DMPresource_set_owner(&object->resource, DMPMAP);
    return true;
#else
    return false;
#endif
  }
  static bool tryacquire_serial(T* object, DMPwaiter*) {
    DMPresource_set_owner(&object->resource, DMPMAP);
    return true;
  }
  static void release(T*) {}

  static void update_predictor(DMPresource* r, int oldowner) {
    TraitsBase::update_predictor(r, oldowner);
  }
  static int predict_next(DMPresource* r) {
    return TraitsBase::predict_next(r);
  }

  static const bool nest_globally        = TraitsBase::nest_globally;
  static const bool acquire_ends_quantum = TraitsBase::acquire_ends_quantum;
  static const bool release_ends_quantum = TraitsBase::release_ends_quantum;
};

template <typename TraitsBase>
void DMPresource_take_ownership(DMPresource* const r) {
  typename TraitsBase::T* object =
      RESOURCE_CONTAINER(r, typename TraitsBase::T);
  DMPresource_acquire<DmpResourceTakeOwnershipTraits<TraitsBase>>(r, object);
}

template <typename Traits>
void DMPresource_release(DMPresource* const r, typename Traits::T* const object,
                         const bool needs_ownership) {
#if defined(DMP_ENABLE_MODE_B_S) || defined(DMP_ENABLE_MODE_OB_S)
  // Don't need to take_ownership when releasing mutex locks.
  // Consider the possible races:
  //   -- release w/ acquire: acquire cannot happen until the next commit anyway
  //   -- release w/ release: two releases in the same round are harmless
  //   (release is idempotent: "just write 0")
  // HOWEVER,
  //   -- release w/ release: two releases in different rounds are bad, e.g.:
  //        1) T1 calls unlock(L) in round #4
  //        2) T2 calls unlock(L) in round #5
  //        3) T3 calls lock(L)   in round #5
  //   -- Because of the round #5 race, T2 can nondeterministically deny T3 the
  //   mutex in quantum 5.
  //   -- Thus, we must take_ownership if we're not the current holder of the
  //   mutex.
  if (!needs_ownership) {
#ifdef DMP_ENABLE_WB_HBSYNC
    r->lastRoundUsed = DMProundNumber;
#endif
    Traits::release(object);
  } else
#endif
  {
    // In the common case, we'll already have ownership here.
    DMPresource_take_ownership<Traits>(r);
    Traits::release(object);
    DMPresource_handoff<Traits>(r);
  }

  // Release ends the quantum?
  if (Traits::release_ends_quantum) {
    DMP_waitForNextQuantum();
  }

#ifdef DMP_ENABLE_TINY_SERIAL_MODE
  if (Traits::nest_globally) {
    if (--DMPMAP->resourceNesting == 0)
      DMP_waitForNextQuantum();
  }
#endif
}

struct DmpResourceWaitUntilWokenTraits {
  typedef void* T;
  static bool tryacquire_parallel(T*, DMPwaiter* w) { return !w->waiting; }
  static bool tryacquire_serial(T*, DMPwaiter* w) { return !w->waiting; }
  static void release(T*) {}

  static void update_predictor(DMPresource* r, int oldowner) {}
  static int predict_next(DMPresource* r) { return -1; }

  static const bool nest_globally        = false;
  static const bool acquire_ends_quantum = false;
  static const bool release_ends_quantum = false;
};

inline void DMPresource_wait_until_woken(DMPresource* const r,
                                         DMPwaiter* const waiter) {
  DMP_ASSERT(waiter);
  // NOTE: waiter->waiting could be false if we've already been signaled

  while (true) {
#ifdef DMP_ENABLE_WB_HBSYNC
    if (!waiter->waiting && waiter->roundReleased < DMProundNumber)
      return;
#else
    if (DMPresource_tryacquire<DmpResourceWaitUntilWokenTraits>(r, NULL,
                                                                waiter))
      return;
#endif

#ifdef DMP_ENABLE_QUANTUM_TIMING
    DMPQuantumEndedWithAcquire = true;
#endif
    DMP_waitForNextQuantum();
  }
}

//--------------------------------------------------------------
// Simplify a common traits pattern
//--------------------------------------------------------------

template <typename T, bool tryacquire(T*, DMPwaiter*)>
struct DmpResourceTryacquireWrapper {
  // Check ownership in parallel mode.
  static bool tryacquire_parallel(T* object, DMPwaiter* w) {
#ifdef DMP_ENABLE_WB_HBSYNC
    // In this case we've guaranteed that it's our deterministic turn.
    if (tryacquire(object, w)) {
      DMPresource_set_owner(&object->resource, DMPMAP);
      return true;
    }
#else
    if (DMPresource_owner(&object->resource) == DMPMAP->threadID) {
      if (tryacquire(object, w))
        return true;
    }
#endif
    return false;
  }

  // Steal ownership in serial mode.
  static bool tryacquire_serial(T* object, DMPwaiter* w) {
    if (tryacquire(object, w)) {
      DMPresource_set_owner(&object->resource, DMPMAP);
      return true;
    }
    return false;
  }
};

//--------------------------------------------------------------
// Predictors
// The interface to a predictor is:
//
// update()
//   -- Called when the current thread acquires resoure 'r'
//
// predict()
//   -- Called to predict the next thread to acquire 'r'
//
// predict_ignoring()
//   -- Like predict, but does not predict that 'ignore' will
//      be the next thread (useful when we know that some other
//      thread must acquire the resource before the given thread
//      can make progress, e.g., in DMPcondvar_wait).
//   -- Use 'ignore = NULL' to act like predict().
//--------------------------------------------------------------

struct DmpEmptyPredictor {
  static void update(DMPresource* r, const int oldowner) {}
  static DmpThreadInfo* predict(DMPresource* r) { return NULL; }
  static DmpThreadInfo* predict_ignoring(DMPresource* r,
                                         DmpThreadInfo* ignore) {
    return NULL;
  }
};

template <int numslots>
struct DmpCoutingPredictorBase {
  // Simple counting predictor: pick the most frequent thread from
  // 'recent', ignoring 'ignore' if 'ignore != NULL'.  Break ties
  // using the most recent thread.

  static DmpThreadInfo* predict_ignoring(short recent[numslots], short nextSlot,
                                         DmpThreadInfo* ignore) {
    struct {
      short threadID;
      short count;
    } counts[numslots];

    // Count the frequency of each thread in 'recent[]'.
    int unique = 0;
    int i, k;
    for (i = 0; i < numslots; ++i) {
      if (recent[i] < 0)
        continue;
      for (k = 0; k < unique; ++k) {
        if (counts[k].threadID == recent[i]) {
          counts[k].count++;
          break;
        }
      }
      if (k == unique) {
        counts[unique].threadID = recent[i];
        counts[unique].count    = 0;
        unique++;
      }
    }

    // Most recent slot.
    const int lastSlot = (nextSlot > 0) ? (nextSlot - 1) : (numslots - 1);

    // Find the most frequent thread in 'recent[]'.
    k = -1;
    for (i = 0; i < unique; ++i) {
      if (ignore != NULL && counts[i].threadID == ignore->threadID)
        continue;
      if (k < 0 || counts[i].count > counts[k].count)
        k = i;
      else if (counts[i].count == counts[k].count &&
               counts[i].threadID == recent[lastSlot])
        k = i;
    }

    if (k >= 0)
      return DMPthreadInfos[counts[k].threadID];
    else
      return NULL;
  }
};

#ifdef DMP_ENABLE_PREDICT_HANDOFF_WINDOWED

struct DmpWindowedPredictor {
  // This predictor using a rolling window to capture the N most
  // recent acquires, and predict that future acquires will be
  // made by the most frequent thread in that window.

  static void update(DMPresource* r, const int oldowner) {
    r->recentAcquires[r->recentAcquiresSlot] = DMPMAP->threadID;
    if (++r->recentAcquiresSlot == ARRAY_SIZE(r->recentAcquires))
      r->recentAcquiresSlot = 0;
  }

  static DmpThreadInfo* predict(DMPresource* r) {
    return predict_ignoring(r, NULL);
  }

  static DmpThreadInfo* predict_ignoring(DMPresource* r,
                                         DmpThreadInfo* ignore) {
    return DmpCoutingPredictorBase<ARRAY_SIZE(
        r->recentAcquires)>::predict_ignoring(r->recentAcquires,
                                              r->recentAcquiresSlot, ignore);
  }
};

#endif // DMP_ENABLE_PREDICT_HANDOFF_WINDOWED

#ifdef DMP_ENABLE_PREDICT_HANDOFF_MARKOV

struct DmpMarkovPredictor {
  // This is a simple markov predictor: we track up to N threads,
  // and for each thread, up to M most recent acquires that follow
  // one of those N threads, and use that to form a simple markov
  // model.

  static void update(DMPresource* r, const int oldowner) {
    for (int i = 0; i < ARRAY_SIZE(r->acquires); ++i) {
      DMPresource::Acquire* a = r->acquires + i;
      if (a->threadID < 0) {
        a->threadID            = oldowner;
        a->recentFollowers[0]  = DMPMAP->threadID;
        a->recentFollowersSlot = 1;
        break;
      }
      if (a->threadID == oldowner) {
        a->recentFollowers[a->recentFollowersSlot] = DMPMAP->threadID;
        if (++a->recentFollowersSlot == ARRAY_SIZE(a->recentFollowers))
          a->recentFollowersSlot = 0;
        break;
      }
    }
  }

  static DmpThreadInfo* predict(DMPresource* r) {
    return predict_ignoring(r, NULL);
  }

  static DmpThreadInfo* predict_ignoring(DMPresource* r,
                                         DmpThreadInfo* ignore) {
    for (int i = 0; i < ARRAY_SIZE(r->acquires); ++i) {
      DMPresource::Acquire* a = r->acquires + i;
      if (a->threadID == DMPMAP->threadID) {
        return DmpCoutingPredictorBase<ARRAY_SIZE(
            a->recentFollowers)>::predict_ignoring(a->recentFollowers,
                                                   a->recentFollowersSlot,
                                                   ignore);
      }
    }
    return NULL;
  }
};

#endif // DMP_ENABLE_PREDICT_HANDOFF_MARKOV

#if defined(DMP_ENABLE_PREDICT_HANDOFF_WINDOWED)
typedef DmpWindowedPredictor DmpDefaultPredictor;
#elif defined(DMP_ENABLE_PREDICT_HANDOFF_MARKOV)
typedef DmpMarkovPredictor DmpDefaultPredictor;
#else
typedef DmpEmptyPredictor DmpDefaultPredictor;
#endif

#endif // _DMP_INTERNAL_RESOURCE_H_
