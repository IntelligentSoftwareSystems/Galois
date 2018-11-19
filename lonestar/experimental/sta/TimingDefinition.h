#ifndef GALOIS_EDA_TIMING_DEFINITION_H
#define GALOIS_EDA_TIMING_DEFINITION_H

#include <string>

#define GALOIS_EDA_USE_DOUBLE_AS_MY_FLOAT 1

#ifdef GALOIS_EDA_USE_DOUBLE_AS_MY_FLOAT
  using MyFloat = double;
#else
  using MyFloat = float;
#endif

enum TimingMode {
  MAX_DELAY_MODE = 0,
  MIN_DELAY_MODE,
};

enum TimingPropAlgo {
  ALGO_TOPO_BARRIER = 0,
  ALGO_BY_DEPENDENCY,
  ALGO_UNORDERED,
  ALGO_TOPO_SOFT_PRIORITY,
};

MyFloat getMyFloat(std::string& str);

#endif // GALOIS_EDA_TIMING_DEFINITION_H
