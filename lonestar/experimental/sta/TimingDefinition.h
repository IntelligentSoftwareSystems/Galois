#ifndef GALOIS_EDA_TIMING_DEFINITION_H
#define GALOIS_EDA_TIMING_DEFINITION_H

#include <string>

using MyFloat = double;

enum TimingMode {
  MAX_DELAY_MODE = 0,
  MIN_DELAY_MODE,
};

MyFloat getMyFloat(std::string& str);

#endif // GALOIS_EDA_TIMING_DEFINITION_H
