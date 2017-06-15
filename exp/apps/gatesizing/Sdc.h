#include <string>

#ifndef GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H
#define GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H

struct SDC {
  float delay;

  SDC(std::string inName);
  ~SDC();

  void printSdcDebug();
};

#endif // GALOIS_SYNOPSIS_DESIGN_CONSTRAINTS_H

