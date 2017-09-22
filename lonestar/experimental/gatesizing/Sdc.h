#include <string>

#include "CellLib.h"
#include "Verilog.h"
#include "CircuitGraph.h"

#ifndef GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H
#define GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H

struct SDC {
  CellLib *cellLib;
  VerilogModule *vModule;
  CircuitGraph *graph;

  float targetDelay;

  SDC(CellLib *lib, VerilogModule *m, CircuitGraph *g);
  ~SDC();

  void setConstraints(std::string inName);
};

#endif // GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H
