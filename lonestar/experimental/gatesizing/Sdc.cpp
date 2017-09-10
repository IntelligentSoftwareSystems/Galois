#include "Sdc.h"
#include "FileReader.h"

#include <string>
#include <iostream>
#include <limits>

static void createClock(FileReader& fRd, SDC *sdc) {
  fRd.nextToken(); // get "-period"
  sdc->targetDelay = std::stof(fRd.nextToken());
  fRd.nextToken(); // get "-name"
  fRd.nextToken(); // get the name for clock port, e.g. clk
  fRd.nextToken(); // get "["
  fRd.nextToken(); // get "get_ports"
  fRd.nextToken(); // get "{"
  fRd.nextToken(); // get "clock"
  fRd.nextToken(); // get "}"
  fRd.nextToken(); // get "]"
}

static void setMaxDelay(FileReader& fRd, SDC *sdc) {
  sdc->targetDelay = std::stof(fRd.nextToken());
  fRd.nextToken(); // get "-from"
  fRd.nextToken(); // get "["
  fRd.nextToken(); // get name of pseudo primary input, e.g. all_inputs, all_registers
  fRd.nextToken(); // get "]"
  fRd.nextToken(); // get "-to"
  fRd.nextToken(); // get "["
  fRd.nextToken(); // get the name of pseudo primary output, e.g. all_outputs, all_registers
  fRd.nextToken(); // get "]"
}

static void setPrimaryInputSlew(FileReader& fRd, SDC *sdc) {
  std::string token = fRd.nextToken();
  if ("-input_transition_fall" == token) {
    sdc->primaryInputFallSlew = std::stof(fRd.nextToken());
  }
  else if ("-input_transition_rise" == token) {
    sdc->primaryInputRiseSlew = std::stof(fRd.nextToken());
  }
  else {
    fRd.pushToken(token);
  }
}

static void setDrivingCell(FileReader& fRd, SDC *sdc) {
  fRd.nextToken(); // get "-lib_cell"
  std::string gateName = fRd.nextToken();
  fRd.nextToken(); // get "-pin"
  std::string pinName = fRd.nextToken();
  fRd.nextToken(); // get "["
  fRd.nextToken(); // get name of pseudo primary input, e.g. all_inputs
  fRd.nextToken(); // get "]"
  setPrimaryInputSlew(fRd, sdc);
  setPrimaryInputSlew(fRd, sdc);
}

static void setLoad(FileReader& fRd, SDC *sdc) {
  fRd.nextToken(); // get "-pin_load"
  sdc->primaryOutputTotalPinC = std::stof(fRd.nextToken());
  fRd.nextToken(); // get "["
  fRd.nextToken(); // get name of pseudo primary output, e.g. all_outputs
  fRd.nextToken(); // get "]"
}

static void readSDC(FileReader& fRd, SDC *sdc) {
  for (std::string token = fRd.nextToken(); token != ""; token = fRd.nextToken()) {
    if ("create_clock" == token) {
      createClock(fRd, sdc);
    }
    else if ("set_max_delay" == token) {
      setMaxDelay(fRd, sdc);
    }
    else if ("set_driving_cell" == token) {
      setDrivingCell(fRd, sdc);
    }
    else if ("set_load") {
      setLoad(fRd, sdc);
    }
  }
}

void SDC::read(std::string inName, CellLib *lib)
{
  cellLib = lib;

  // default values
  targetDelay = std::numeric_limits<float>::infinity();
  primaryInputRiseSlew = cellLib->cells.at("INV_X4")->outPins.at("ZN")->cellRise.at("A")->index[0][3];
  primaryInputFallSlew = cellLib->cells.at("INV_X4")->outPins.at("ZN")->cellFall.at("A")->index[0][3];
  primaryOutputTotalPinC = 2.0 * cellLib->cells.at("INV_X1")->inPins.at("A")->capacitance;
  primaryOutputTotalNetC = cellLib->defaultWireLoad->wireCapacitance(1);

  if (inName.empty()) {
    std::cout << "No .sdc specified. Use the default values.\n";
  }
  else {
    std::cout << "Use the values in " << inName << ".\n";

    char delimiters[] = {
      '(', ')',
      ',', ':', ';', 
      '/',
      '#',
      '[', ']', 
      '{', '}',
      '*',
      '\"', '\\'
    };

    char separators[] = {
      ' ', '\t', '\n', ','
    };

    FileReader fRd(inName, delimiters, sizeof(delimiters), separators, sizeof(separators));
    readSDC(fRd, this);
  }
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
  std::cout << "primary input rise slew = " << primaryInputRiseSlew << std::endl;
  std::cout << "primary input fall slew = " << primaryInputFallSlew << std::endl;
  std::cout << "primary output pin capacitance = " << primaryOutputTotalPinC << std::endl;
  std::cout << "primary output net capacitance = " << primaryOutputTotalNetC << std::endl;
}

