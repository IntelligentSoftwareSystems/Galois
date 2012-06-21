#include "Galois/Runtime/Sampling.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/ll/EnvCheck.h"
#include <cstdlib>

static void endPeriod() {
  int val;
  if (GaloisRuntime::LL::EnvCheck("GALOIS_EXIT_AFTER_SAMPLING", val)) {
    exit(val);
  }
}

static void beginPeriod() {
  int val;
  if (GaloisRuntime::LL::EnvCheck("GALOIS_EXIT_BEFORE_SAMPLING", val)) {
    exit(val);
  }
}

#ifdef GALOIS_USE_VTUNE
#include "ittnotify.h"
#include "Galois/Runtime/ll/TID.h"

static bool isOn;

void GaloisRuntime::beginSampling() {
  beginPeriod();
  if (!isOn && LL::getTID() == 0)
    __itt_resume();
  isOn = true;
}

void GaloisRuntime::endSampling() {
  if (isOn && LL::getTID() == 0)
    __itt_pause();
  isOn = false;
  endPeriod();
}

#else

void GaloisRuntime::beginSampling() { beginPeriod(); }
void GaloisRuntime::endSampling() { endPeriod(); }

#endif
