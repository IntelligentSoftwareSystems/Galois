#include <unordered_set>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#ifndef GALOIS_CELLLIB_H
#define GALOIS_CELLLIB_H

enum LutVar {
  LUT_VAR_INPUT_NET_TRANSITION,
  LUT_VAR_TOTAL_OUTPUT_NET_CAPACITANCE,
  LUT_VAR_CONSTRAINED_PIN_TRANSITION,
  LUT_VAR_RELATED_PIN_TRANSITION,
  LUT_VAR_UNDEFINED
};

enum TimingSense {
  TIMING_SENSE_POSITIVE_UNATE,
  TIMING_SENSE_NEGATIVE_UNATE,
  TIMING_SENSE_NON_UNATE,
  TIMING_SENSE_UNDEFINED
};

enum PinType {
  PIN_OUTPUT,
  PIN_INPUT,
  PIN_INTERNAL,
  PIN_UNDEFINED
};

struct LutTemplate {
  std::string name;
  std::vector<LutVar> var;
  std::vector<size_t> dim;
};

struct LUT {
  LutTemplate *lutTemplate;
  std::vector<std::vector<float> > index, value;
};

struct Cell;

struct CellPin {
  std::string name;
  float capacitance;
  TimingSense tSense;
  PinType pinType;
  Cell *cell;
  std::unordered_map<std::string, LUT *> cellRise, cellFall, riseTransition, fallTransition;
  std::unordered_map<std::string, LUT *> risePower, fallPower;
};

struct Cell {
  std::string name, familyName;
  float area, cellLeakagePower;
  size_t driveStrength;
  std::unordered_map<std::string, CellPin *> outPins, inPins, internalPins, cellPins;
};

struct WireLoad {
  std::string name;
  float capacitance, resistance, slope;
  std::map<size_t, float> fanoutLength;
};

struct CellLib {
  typedef std::unordered_map<std::string, Cell *> CellMap;

  std::string name;
  CellMap cells;
  std::unordered_map<std::string, CellMap> cellFamilies;
  std::unordered_map<std::string, LutTemplate *> lutTemplates;
  std::unordered_map<std::string, WireLoad *> wireLoads;

  WireLoad *defaultWireLoad;

  CellLib(std::string inName);
  ~CellLib();

  void printCellLibDebug();
};

#endif // GALOIS_CELLLIB_H

