#include "Galois/Runtime/Sampling.h"

#ifdef GALOIS_VTUNE
#include "ittnotify.h"

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
