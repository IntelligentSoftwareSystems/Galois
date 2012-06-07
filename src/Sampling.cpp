#include "Galois/Runtime/Sampling.h"

#ifdef GALOIS_USE_VTUNE
#include "ittnotify.h"
#include "Galois/Runtime/ll/TID.h"

static bool isOn;

void GaloisRuntime::beginSampling() {
  if (!isOn && LL::getTID() == 0)
    __itt_resume();
  isOn = true;
}

void GaloisRuntime::endSampling() {
  if (isOn && LL::getTID() == 0)
    __itt_pause();
  isOn = false;
}

#else

void GaloisRuntime::beginSampling() { }
void GaloisRuntime::endSampling() { }

#endif
