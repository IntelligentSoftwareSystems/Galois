#include <string>

#include "CellLib.h"

#ifndef GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H
#define GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H

struct SDC {
  float targetDelay;
  float primaryInputSlew;
  float primaryOutputCapacitance;

  CellLib& cellLib;

  SDC(std::string inName, CellLib& lib);
  ~SDC();

  void printSdcDebug();
};

#endif // GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H

