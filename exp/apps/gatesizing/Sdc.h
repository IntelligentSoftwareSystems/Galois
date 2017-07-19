#include <string>

#include "CellLib.h"

#ifndef GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H
#define GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H

struct SDC {
  float targetDelay;
  float primaryInputSlew;
  float primaryOutputTotalNetC;

  CellLib *cellLib;

  SDC();
  ~SDC();

  void read(std::string inName, CellLib *lib);
  void clear();
  void printSdcDebug();
};

#endif // GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H

