#include "Sdc.h"

#include <fstream>
#include <iostream>

SDC::SDC(std::string inName) {
  delay = 0.0;
  if (!inName.empty()) {
    std::ifstream ifs(inName);
    if (ifs.is_open()) {
      std::string s1, s2;
      ifs >> s1 >> s2 >> delay;
    } else {
      std::cout << "Cannot open " << inName << ". Set delay = 0.0" << std::endl;
    }
  }
}

SDC::~SDC() {
}

void SDC::printSdcDebug() {
  std::cout << "delay = " << delay << std::endl;
}

