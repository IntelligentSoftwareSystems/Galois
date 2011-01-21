#include "Galois/Runtime/Support.h"

#include <iostream>

void GaloisRuntime::reportStat(const char* text, unsigned long val) {
  std::cerr << "STAT: " << text << " " << val << "\n";
}

void GaloisRuntime::reportStat(const char* text, double val) {
  std::cerr << "STAT: " << text << " " << val << "\n";
}

//Report Warnings
void GaloisRuntime::reportWarning(const char* text) {
  std::cerr << "WARNING: " << text << "\n";
}
