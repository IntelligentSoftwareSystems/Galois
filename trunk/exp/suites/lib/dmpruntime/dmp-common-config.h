// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// This is the set of #defines which can be enabled.
// All compile-time opts start with DMP_ENABLE_*, so they can be spotted easy.
//

#ifndef _DMP_COMMON_CONFIG_H_
#define _DMP_COMMON_CONFIG_H_

#include "config.h"

//--------------------------------------------------------------
// Execution Models
//--------------------------------------------------------------

// #define DMP_ENABLE_MODEL_O_S    ::  O|S
// #define DMP_ENABLE_MODEL_B_S    ::  B|S
// #define DMP_ENABLE_MODEL_OB_S   ::  OB|S
// #define DMP_ENABLE_MODEL_O_B_S  ::  O|B|S
// #define DMP_ENABLE_MODEL_STM    ::  STM (experiment)

// ----
// Options for ownership modes
//
// For more accurate single-threaded benchmarks
//   #define DMP_DISABLE_SINGLE_THREADED_ALWAYS_SHARE

// ----
// Options for buffered modes
//
// Move-to-front buffer optimization
//   #define DMP_ENABLE_WB_MOVE_TO_FRONT
//
// Commit modes
//   #define DMP_ENABLE_WB_NONDET_COMMIT
//   #define DMP_ENABLE_WB_PARALLEL_COMMIT
//
// Quantum building
//   #define DMP_ENABLE_WB_THREADLOCAL_SYNCOPS
//
// Hack around an LLVM bug
//   #define DMP_ENABLE_WB_BAD_ALIGNMENTS
//
// Happens-before consistency
//   #define DMP_ENABLE_WB_HBSYNC
//   #define DMP_ENABLE_WB_HBSYNC_UNLOCKOPT

// ----
// Options for O|B|S only
//
// Log the readset while buffering and use for ownership policies
//   #define DMP_ENABLE_WB_READLOG

// ----
// Options for O|S only
//
// Handoff features
//   #define DMP_ENABLE_SLOW_HANDOFF
//   #define DMP_ENABLE_FAST_HANDOFF
//   #define DMP_ENABLE_FAST_HANDOFF_QUANTUM_OPT
//   #define DMP_ENABLE_PREDICT_HANDOFF_WINDOWED
//   #define DMP_ENABLE_PREDICT_HANDOFF_MARKOV
//   #define DMP_ENABLE_PREDICT_HANDOFF_BARRIER
//
// Data grouping features
//   #define DMP_ENABLE_DATA_GROUP_BY_MUTEX

//--------------------------------------------------------------
// Common Options
//--------------------------------------------------------------

// Quantum building features
// #define DMP_ENABLE_EMPTY_SERIAL_MODE
// #define DMP_ENABLE_TINY_SERIAL_MODE
// #define DMP_ENABLE_MUTEX_LOCK_ENDQUANTUM
// #define DMP_ENABLE_MUTEX_UNLOCK_ENDQUANTUM

// Profiling
// #define DMP_ENABLE_INSTRUMENT_ACQUIRES
// #define DMP_ENABLE_INSTRUMENT_WORK
// #define DMP_ENABLE_QUANTUM_TIMING
// #define DMP_ENABLE_ROUND_TIMING

// Debugging
// #define DMP_ENABLE_DEBUGGING 

// For non-libhoard builds
// #define DMP_ENABLE_LIBHOARD

//--------------------------------------------------------------
// Implicit Options (don't define these manually!)
//--------------------------------------------------------------

#if defined(DMP_ENABLE_MODEL_O_S)   ||\
    defined(DMP_ENABLE_MODEL_O_B_S) ||\
    defined(DMP_ENABLE_MODEL_STM)
#define DMP_ENABLE_OWNERSHIP_MODE
#endif

#if defined(DMP_ENABLE_MODEL_B_S)  ||\
    defined(DMP_ENABLE_MODEL_OB_S) ||\
    defined(DMP_ENABLE_MODEL_O_B_S)
#define DMP_ENABLE_BUFFERED_MODE
#endif

#if defined(DMP_ENABLE_WB_HBSYNC_UNLOCKOPT)
#define DMP_ENABLE_WB_HBSYNC
#endif

//--------------------------------------------------------------
// Constraints
//--------------------------------------------------------------

#if !defined(DMP_ENABLE_MODEL_O_S)   &&\
    !defined(DMP_ENABLE_MODEL_B_S)   &&\
    !defined(DMP_ENABLE_MODEL_OB_S)  &&\
    !defined(DMP_ENABLE_MODEL_O_B_S) &&\
    !defined(DMP_ENABLE_MODEL_STM)
#error "No execution model specified!"
#endif

#if !defined(DMP_ENABLE_MODEL_O_S) &&\
    (defined(DMP_ENABLE_SLOW_HANDOFF) ||\
     defined(DMP_ENABLE_FAST_HANDOFF) ||\
     defined(DMP_ENABLE_FAST_HANDOFF_QUANTUM_OPT) ||\
     defined(DMP_ENABLE_PREDICT_HANDOFF_WINDOWED) ||\
     defined(DMP_ENABLE_PREDICT_HANDOFF_MARKOV)   ||\
     defined(DMP_ENABLE_PREDICT_HANDOFF_BARRIER)  ||\
     defined(DMP_ENABLE_DATA_GROUP_BY_MUTEX))
#error "Options require DMP_ENABLE_MODEL_O_S"
#endif

#if !defined(DMP_ENABLE_BUFFERED_MODE) &&\
    (defined(DMP_ENABLE_WB_MOVE_TO_FRONT) ||\
     defined(DMP_ENABLE_WB_NONDET_COMMIT)  ||\
     defined(DMP_ENABLE_WB_PARALLEL_COMMIT)  ||\
     defined(DMP_ENABLE_WB_THREADLOCAL_SYNCOPS)  ||\
     defined(DMP_ENABLE_WB_BAD_ALIGNMENTS)  ||\
     defined(DMP_ENABLE_WB_HBSYNC))
#error "Options require buffering"
#endif

#if defined(DMP_ENABLE_WB_HB_SYNC) &&\
   !defined(DMP_ENABLE_MODEL_B_S)  &&\
   !defined(DMP_ENABLE_MODEL_OB_S)
#error "Option requires DMP_ENABLE_MODEL_{B,OB}_S"
#endif

#if !defined(DMP_ENABLE_MODEL_O_B_S) && defined(DMP_ENABLE_BUFFERED_READLOG)
#error "Option requires DMP_ENABLE_MODEL_O_B_S"
#endif

#if defined(DMP_ENABLE_WB_PARALLEL_COMMIT) && defined(DMP_ENABLE_WB_NONDET_COMMIT)
#error "Parallel commit and nondet commit are mutually exclusive"
#endif

#if defined(DMP_ENABLE_WB_THREADLOCAL_SYNCOPS) && defined(DMP_ENABLE_WB_HB_SYNC)
#error "Threadlocal sync and happens-before sync are mutually exclusive"
#endif

#if defined(DMP_ENABLE_EMPTY_SERIAL_MODE)
#error "EmptySeriaMode is broken and probably won't help anyway"
#endif

//--------------------------------------------------------------
// Obsolete
//--------------------------------------------------------------

#ifdef DMP_SINGLE_THREAD_OPT
#error "DMP_SINGLE_THREAD_OPT is not implemented anymore"
#endif
#ifdef DMP_UNLIMITED_THREADS
#error "DMP_UNLIMITED_THREADS is now the default"
#endif
#ifdef DMP_ENABLE_SLOW_HANDOFF_MUTEX
#error "use DMP_ENABLE_SLOW_HANDOFF instead"
#endif
#ifdef DMP_ENABLE_FAST_HANDOFF_MUTEX
#error "use DMP_ENABLE_FAST_HANDOFF instead"
#endif
// Obsolete: renamed to DMP_ENABLE_* / DMP_DISABLE_*
#ifdef DMP_NO_SINGLE_THREADED_ALWAYS_SHARE
#error "use DMP_DISABLE_SINGLE_THREADED_ALWAYS_SHARE"
#endif
#ifdef DMP_INSTRUMENTATION
#error "use DMP_ENABLE_INSTRUMENT_*"
#endif
#ifdef DMP_ENABLE_INSTRUMENTATION
#error "use DMP_ENABLE_INSTRUMENT_*"
#endif
#ifdef DMP_QUANTUM_TIMING
#error "use DMP_ENABLE_QUANTUM_TIMING"
#endif
#ifdef ENABLE_DMP_DEBUGGING
#error "use DMP_ENABLE_DEBUGGING"
#endif
#ifdef DMP_ENABLE_NONLOCAL_OWNER_DETECTION
#error "the nonlocal bit is no longer implemented"
#endif
#ifdef DMP_ENABLE_NONDET_WB_COMMIT
#error "use DMP_ENABLE_WB_NONDET_COMMIT"
#endif
#ifdef DMP_ENABLE_BUFFERED_READLOG
#error "use DMP_ENABLE_WB_READLOG"
#endif

#endif // _DMP_COMMON_CONFIG_H_
