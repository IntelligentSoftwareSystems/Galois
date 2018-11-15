#ifndef GALOIS_EDA_CLOCK_H
#define GALOIS_EDA_CLOCK_H

#include <iostream>
#include <vector>

#include "TimingDefinition.h"
#include "Verilog.h"

struct ClockEdge {
  MyFloat t;
  bool isRise;

public:
  void print(std::ostream& os);
};

struct Clock {
  MyFloat period;
  std::vector<ClockEdge> waveform;
  VerilogPin* src;
  std::string name;

public:
  void print(std::ostream& os);
};

#endif // GALOIS_EDA_CLOCK_H
