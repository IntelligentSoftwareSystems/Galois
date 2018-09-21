#ifndef GALOIS_EDA_TIMING_DEFINITION_H
#define GALOIS_EDA_TIMING_DEFINITION_H

#include <string>

using MyFloat = double;

enum TimingMode {
  TIMING_MODE_MAX_DELAY = 0,
  TIMING_MODE_MIN_DELAY,
};

MyFloat getMyFloat(std::string& str);

#endif // GALOIS_EDA_TIMING_DEFINITION_H
