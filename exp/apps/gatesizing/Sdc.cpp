#include "Sdc.h"

#include <fstream>
#include <iostream>

void SDC::read(std::string inName, CellLib *lib)
{
  cellLib = lib;

  targetDelay = 0.0;
  if (!inName.empty()) {
    std::ifstream ifs(inName);
    if (ifs.is_open()) {
      std::string s1, s2;
      ifs >> s1 >> s2 >> targetDelay;
    } else {
      std::cout << "Cannot open " << inName << ". Set targetDelay = 0.0" << std::endl;
    }
  }

  primaryInputSlew = cellLib->cells.at("INV_X4")->outPins.at("ZN")->cellRise.at("A")->index[0][3];
  primaryOutputTotalPinC = 2.0 * cellLib->cells.at("INV_X1")->inPins.at("A")->capacitance;
  primaryOutputTotalNetC = cellLib->defaultWireLoad->wireCapacitance(1);
}

void SDC::clear() {
}

SDC::SDC() {
}

SDC::~SDC() {
  clear();
}

void SDC::printSdcDebug() {
  std::cout << "targetDelay = " << targetDelay << std::endl;
  std::cout << "primary input slew = " << primaryInputSlew << std::endl;
  std::cout << "primary output pin capacitance = " << primaryOutputTotalPinC << std::endl;
  std::cout << "primary output net capacitance = " << primaryOutputTotalNetC << std::endl;
}

