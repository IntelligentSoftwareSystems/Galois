// (C) 2010 University of Washington.
// The software is subject to an academic license.
// Terms are contained in the LICENSE file.
//
// PThread Library - once objects
//

#include "dmp-internal.h"

#define ONCE_DEBUG_MSG(O,msg) DEBUG_MSG( DEBUG_ONCE, \
                                         msg "(%p) @%llu T:%d", \
                                         (O), DMProundNumber, DMPMAP->threadID )

struct DmpOnceTraits {
  typedef DMPonce T;

  // These are unused: DMPresource_take_ownership() uses its own, internally.
  static bool tryacquire_parallel(DMPmutex* mutex, DMPwaiter* w) { return false; }
  static bool tryacquire_serial(DMPmutex* mutex, DMPwaiter* w) { return false; }
  static void release(DMPmutex* mutex) {}

  // No prediction: ownership is only acquired once!
  static void update_predictor(DMPresource* r, int oldowner) {}
  static DmpThreadInfo* predict_next(DMPresource* r) { return NULL; }

  static const bool nest_globally = false;
  static const bool acquire_ends_quantum = false;
  static const bool release_ends_quantum = false;
};

//--------------------------------------------------------------
// API
//--------------------------------------------------------------

int DMPonce_once(DMPonce* once, void (*init_routine)(void)) {
  // Common case: already initialized (reads here are safe).
  if (once->done)
    return 0;
  // Rare case: not initialized yet.
  // We need to double-check for initialization after taking ownership.
  DMPresource_take_ownership<DmpOnceTraits>(&once->resource);
  if (!once->done) {
    once->done = 1;
    (*init_routine)();
  }
  return 0;
}
