// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// Shared interface for handoffable resources.
// Include from dmp-runtime-common.h only!
//

#ifndef _DMP_COMMON_RESOURCE_H_
#define _DMP_COMMON_RESOURCE_H_

#ifdef DMP_ENABLE_HANDOFF
#error "don't define DMP_ENABLE_HANDOFF manually"
#endif
#ifdef DMP_ENABLE_DATA_GROUPING
#error "don't define DMP_ENABLE_DATA_GROUPING manually"
#endif

#if defined(DMP_ENABLE_SLOW_HANDOFF) || defined(DMP_ENABLE_FAST_HANDOFF)
#define DMP_ENABLE_HANDOFF
#endif

#if defined(DMP_ENABLE_DATA_GROUP_BY_MUTEX)
#define DMP_ENABLE_DATA_GROUPING
#endif

// Handoff prediction makes no sense without handoff
#if (defined(DMP_ENABLE_PREDICT_HANDOFF_WINDOWED) ||                           \
     defined(DMP_ENABLE_PREDICT_HANDOFF_MARKOV) ||                             \
     defined(DMP_ENABLE_PREDICT_HANDOFF_BARRIER)) &&                           \
    !defined(DMP_ENABLE_HANDOFF)
#error "DMP_ENABLE_PREDICT_HANDOFF_* without DMP_ENABLE_*_HANDOFF"
#endif

// Mutually-exclusive prediction modes
#if defined(DMP_ENABLE_PREDICT_HANDOFF_WINDOWED) &&                            \
    defined(DMP_ENABLE_PREDICT_HANDOFF_MARKOV)
#error "multiple prediction methods enabled!"
#endif

//-----------------------------------------------------------------------
// Handoff info: embed into all resources that can be handed-off
//-----------------------------------------------------------------------

struct DmpThreadInfo;
typedef struct DMPresource DMPresource;
typedef struct DMPwaiter DMPwaiter;

struct DMPwaiter {
  volatile uint32_t waiting; // 1 if on the wait queue, 0 if not
  uint32_t rounds;           // number of rounds the thread has been waiting
  struct DmpThreadInfo* dmp; // the waiting thread
  DMPwaiter* next;
#ifdef DMP_ENABLE_WB_HBSYNC
  volatile uint64_t roundReleased; // round at which we were released (must wait
                                   // one more round to wakeup)
#endif
};

struct DMPresource {
  // Resource state
  //   bit  1-16 :: thread ID of the owner (part of a distributed MOT)
  //   bit 17-20 :: resource type
  //   bit 21    :: resource has been used (only set in certain compile modes)
  int state;

  // Linked-list of threads waiting on this resource.
  DMPwaiter* waiters;

#ifdef DMP_ENABLE_WB_HBSYNC
  // Last quantum this resource was used by the current owner.
  volatile uint64_t lastRoundUsed;
#endif

#ifdef DMP_ENABLE_DATA_GROUPING
  // Points up the chain of resources held by 'owner', if any.
  DMPresource* outer;
#endif

#ifdef DMP_ENABLE_PREDICT_HANDOFF_WINDOWED
  // Simple windowed predictor
  short recentAcquires[4]; // last N threads to acquire ('-1' for empty)
  int recentAcquiresSlot;  // next slot in recentAcquires[]
#endif

#ifdef DMP_ENABLE_PREDICT_HANDOFF_MARKOV
  struct Acquire {
    short threadID;            // thread to acquire ('-1' for not used yet)
    short recentFollowersSlot; // next slot in recentFollowers[]
    short recentFollowers[4];  // last M threads to follow 'acqure'
  } acquires[4];               // first N threads to acquire
#endif

#ifdef DMP_ENABLE_INSTRUMENT_ACQUIRES
  uint64_t lastRoundAcquired;
#endif
};

#define DMP_RESOURCE_STATE_OWNER_MASK ((1 << 16) - 1)
#define DMP_RESOURCE_STATE_TYPE_MASK (0xf << 16) // max of 8 types (0-7)
#define DMP_RESOURCE_STATE_USED (1 << 21)

#define DMP_RESOURCE_TYPE_ONCE (0 << 16)
#define DMP_RESOURCE_TYPE_MUTEX (1 << 16)
#define DMP_RESOURCE_TYPE_RWLOCK (2 << 16)
#define DMP_RESOURCE_TYPE_SEM (3 << 16)
#define DMP_RESOURCE_TYPE_CONDVAR (4 << 16)
#define DMP_RESOURCE_TYPE_BARRIER (5 << 16)

// Ugly ugly field initializers.
// These can't use C's named field initializers since those break in C++.
#define DMP_RESOURCE_INITIALIZER_DATA_GROUPING
#define DMP_RESOURCE_INITIALIZER_HANDOFF_WINDOWED
#define DMP_RESOURCE_INITIALIZER_HANDOFF_MARKOV
#define DMP_RESOURCE_INITIALIZER_INSTRUMENT_ACQUIRES

#ifdef DMP_ENABLE_DATA_GROUPING
#undef DMP_RESOURCE_INITIALIZER_DATA_GROUPING
#define DMP_RESOURCE_INITIALIZER_DATA_GROUPING /*.outer =*/NULL,
#endif

#ifdef DMP_ENABLE_PREDICT_HANDOFF_WINDOWED
#undef DMP_RESOURCE_INITIALIZER_HANDOFF_WINDOWED
#define DMP_RESOURCE_INITIALIZER_HANDOFF_WINDOWED                              \
  /*.recentAcquires =*/{-1, -1, -1, -1}, /*.recentAcquiresSlot =*/0,
#endif

#ifdef DMP_ENABLE_PREDICT_HANDOFF_MARKOV
#undef DMP_RESOURCE_INITIALIZER_HANDOFF_MARKOV
#define DMP_RESOURCE_INITIALIZER_ACQUIRE                                       \
  { /*.threadID =*/                                                            \
    -1, /*.recentFollowersSlot =*/0, /*.recentFollowers =*/{                   \
      -1, -1, -1, -1                                                           \
    }                                                                          \
  }
#define DMP_RESOURCE_INITIALIZER_HANDOFF_MARKOV                                \
  /*.acquires =*/{                                                             \
      DMP_RESOURCE_INITIALIZER_ACQUIRE, DMP_RESOURCE_INITIALIZER_ACQUIRE,      \
      DMP_RESOURCE_INITIALIZER_ACQUIRE, DMP_RESOURCE_INITIALIZER_ACQUIRE},
#endif

#ifdef DMP_ENABLE_INSTRUMENT_ACQUIRES
#undef DMP_RESOURCE_INITIALIZER_INSTRUMENT_ACQUIRES
#define DMP_RESOURCE_INITIALIZER_INSTRUMENT_ACQUIRES /*.lastRoundAcquired =*/-1,
#endif

#define DMP_RESOURCE_INITIALIZER_FIELDS                                        \
  /*.waiters =*/NULL, DMP_RESOURCE_INITIALIZER_DATA_GROUPING                   \
                          DMP_RESOURCE_INITIALIZER_HANDOFF_WINDOWED            \
                              DMP_RESOURCE_INITIALIZER_HANDOFF_MARKOV          \
                                  DMP_RESOURCE_INITIALIZER_INSTRUMENT_ACQUIRES

// Ugly ugly initializers.
#define DMP_WAITER_INIT                                                        \
  { /*.waiting =*/                                                             \
    0, /*.rounds =*/0, /*.dmp =*/NULL, /*.next =*/NULL                         \
  }

#define DMP_RESOURCE_ONCE_INIT                                                 \
  { /*.state =*/                                                               \
    DMP_RESOURCE_TYPE_ONCE, DMP_RESOURCE_INITIALIZER_FIELDS                    \
  }

#define DMP_RESOURCE_MUTEX_INIT                                                \
  { /*.state =*/                                                               \
    DMP_RESOURCE_TYPE_MUTEX, DMP_RESOURCE_INITIALIZER_FIELDS                   \
  }

#define DMP_RESOURCE_RWLOCK_INIT                                               \
  { /*.state =*/                                                               \
    DMP_RESOURCE_TYPE_RWLOCK, DMP_RESOURCE_INITIALIZER_FIELDS                  \
  }

#define DMP_RESOURCE_SEM_INIT                                                  \
  { /*.state =*/                                                               \
    DMP_RESOURCE_TYPE_SEM, DMP_RESOURCE_INITIALIZER_FIELDS                     \
  }

#define DMP_RESOURCE_CONDVAR_INIT                                              \
  { /*.state =*/                                                               \
    DMP_RESOURCE_TYPE_CONDVAR, DMP_RESOURCE_INITIALIZER_FIELDS                 \
  }

#define DMP_RESOURCE_BARRIER_INIT                                              \
  { /*.state =*/                                                               \
    DMP_RESOURCE_TYPE_BARRIER, DMP_RESOURCE_INITIALIZER_FIELDS                 \
  }

#endif // _DMP_COMMON_RESOURCE_H_
