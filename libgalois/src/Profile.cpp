#include "galois/runtime/Profile.h"

#ifdef GALOIS_USE_PAPI
extern "C" {
#include <papi.h>
#include <papiStdEventDefs.h>
}
#include <iostream>

unsigned long galois::runtime::internal::papiGetTID(void) {
  return galois::substrate::ThreadPool::getTID();
}
#endif// GALOIS_USE_PAPI


#if 0
#include "galois/runtime/Profile.h"
#include "galois/runtime/Statistics.h"
#include "galois/substrate/EnvCheck.h"
#include "galois/substrate/ThreadPool.h"
#include "galois/gIO.h"
#include <cstdlib>

static void endPeriod() {
  int val;
  if (galois::substrate::EnvCheck("GALOIS_EXIT_AFTER_SAMPLING", val)) {
    exit(val);
  }
}

static void beginPeriod() {
  int val;
  if (galois::substrate::EnvCheck("GALOIS_EXIT_BEFORE_SAMPLING", val)) {
    exit(val);
  }
}

#ifdef GALOIS_USE_VTUNE
#include "ittnotify.h"

namespace vtune {
static bool isOn;
static void begin() {
  if (!isOn && galois::substrate::ThreadPool::getTID() == 0)
    __itt_resume();
  isOn = true;
  galois::gDebug("vtune sampling started");
}

static void end() {
  if (isOn && galois::substrate::ThreadPool::getTID() == 0)
    __itt_pause();
  isOn = false;
  galois::gDebug("vtune sampling stopped");
}
}
#else
namespace vtune {
static void begin() {}
static void end() {}
}
#endif

#ifdef GALOIS_USE_HPCTOOLKIT
#include <hpctoolkit.h>
#include "galois/runtime/ll/TID.h"

namespace hpctoolkit {
static bool isOn;
static void begin() {
  if (!isOn && galois::substrate::ThreadPool::getTID() == 0)
    hpctoolkit_sampling_start();
  isOn = true;
}

static void end() {
  if (isOn && galois::substrate::ThreadPool::getTID() == 0)
    hpctoolkit_sampling_stop();
  isOn = false;
}
}
#else
namespace hpctoolkit {
static void begin() {}
static void end() {}
}
#endif

#ifdef GALOIS_USE_PAPI
extern "C" {
#include <papi.h>
#include <papiStdEventDefs.h>
}
#include <iostream>

namespace papi {
static bool isInit;
static bool isSampling;
static __thread int papiEventSet = PAPI_NULL;

//static int papiEvents[2] = {PAPI_L3_TCA,PAPI_L3_TCM};
//static const char* papiNames[2] = {"L3_ACCESSES","L3_MISSES"};

static int papiEvents[] = {PAPI_TOT_INS, PAPI_TOT_CYC};
static const char* papiNames[] = {"Instructions", "Cycles" };

//static int papiEvents[2] = {PAPI_L1_DCM, PAPI_TOT_CYC};
//static const char* papiNames[2] = {"L1DCMCounter", "CyclesCounter"};

static_assert(sizeof(papiEvents)/sizeof(*papiEvents) == sizeof(papiNames)/sizeof(*papiNames),
    "PAPI Events != PAPI Names");

static unsigned long galois_get_thread_id() {
  return galois::substrate::ThreadPool::getTID();
}

static void begin(bool mainThread) {
  if (mainThread) {
    if (isSampling)
      GALOIS_DIE("Sampling already begun");
    isSampling = true;
  } else if (!isSampling) {
    return;
  }

  int rv;

  // Init library
  if (!isInit) {
    rv = PAPI_library_init(PAPI_VER_CURRENT);
    if (rv != PAPI_VER_CURRENT && rv < 0) {
      GALOIS_DIE("PAPI library version mismatch!");
    }
    if (rv < 0) GALOIS_DIE(PAPI_strerror(rv));
    if ((rv = PAPI_thread_init(galois_get_thread_id)) != PAPI_OK)
      GALOIS_DIE(PAPI_strerror(rv));
    isInit = true;
  }
  // Register thread
  if ((rv = PAPI_register_thread()) != PAPI_OK) 
    GALOIS_DIE(PAPI_strerror(rv));
  // Create the Event Set
  if ((rv = PAPI_create_eventset(&papiEventSet)) != PAPI_OK)
    GALOIS_DIE(PAPI_strerror(rv));
  if ((rv = PAPI_add_events(papiEventSet, papiEvents, sizeof(papiEvents)/sizeof(*papiEvents))) != PAPI_OK)
    GALOIS_DIE(PAPI_strerror(rv));
  // Start counting events in the event set
  if ((rv = PAPI_start(papiEventSet)) != PAPI_OK)
    GALOIS_DIE(PAPI_strerror(rv));
}

static void end(bool mainThread) {
  if (mainThread) {
    if (!isSampling)
      GALOIS_DIE("Sampling not yet begun");
    isSampling = false;
  } else if (!isSampling) {
    return;
  }

  int rv;

  long_long papiResults[sizeof(papiNames)/sizeof(*papiNames)];

  // Get the values
  if ((rv = PAPI_stop(papiEventSet, papiResults)) != PAPI_OK)
    GALOIS_DIE(PAPI_strerror(rv));
  // Remove all events in the eventset
  if ((rv = PAPI_cleanup_eventset(papiEventSet)) != PAPI_OK)
    GALOIS_DIE(PAPI_strerror(rv));
  // Free all memory and data structures, EventSet must be empty.
  if ((rv = PAPI_destroy_eventset(&papiEventSet)) != PAPI_OK)
    GALOIS_DIE(PAPI_strerror(rv));
  // Unregister thread
  if ((rv = PAPI_unregister_thread()) != PAPI_OK) 
    GALOIS_DIE(PAPI_strerror(rv));

  for (unsigned i = 0; i < sizeof(papiNames)/sizeof(*papiNames); ++i)
    galois::runtime::reportStat_Tsum("PAPI-Prof", papiNames[i], papiResults[i]);

}

}
#else
namespace papi {
static void begin(bool) {}
static void end(bool) {}
}
#endif

void galois::runtime::beginThreadSampling() {
  papi::begin(false);
}

void galois::runtime::endThreadSampling() {
  papi::end(false);
}

void galois::runtime::beginSampling() {
  beginPeriod();
  papi::begin(true);
  vtune::begin();
  hpctoolkit::begin();
}

void galois::runtime::endSampling() {
  hpctoolkit::end();
  vtune::end();
  papi::end(true);
  endPeriod();
}

#endif
