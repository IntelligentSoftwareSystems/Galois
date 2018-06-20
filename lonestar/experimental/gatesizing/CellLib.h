/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include <unordered_set>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <utility>

#include <boost/functional/hash.hpp>

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
  TIMING_SENSE_NON_UNATE,
  TIMING_SENSE_POSITIVE_UNATE,
  TIMING_SENSE_NEGATIVE_UNATE,
  TIMING_SENSE_UNDEFINED
};

enum PinType { PIN_OUTPUT, PIN_INPUT, PIN_INTERNAL, PIN_UNDEFINED };

struct LutTemplate {
  std::string name;
  std::vector<LutVar> var;
  std::vector<size_t> dim;
};

struct LUT {
  LutTemplate* lutTemplate;
  std::vector<std::vector<float>> index, value;

  float lookup(std::vector<float>& param);
};

struct Cell;

struct CellPin {
  typedef std::pair<std::string, std::string> TSenseKey;
  typedef std::unordered_map<TSenseKey, TimingSense, boost::hash<TSenseKey>>
      MapOfTimingSense;

  typedef std::pair<std::string, TimingSense> TableSetKey;
  typedef std::unordered_map<std::string, LUT*> TableSet;
  typedef std::unordered_map<TableSetKey, TableSet, boost::hash<TableSetKey>>
      MapOfTableSet;

  std::string name;
  float riseCapacitance, fallCapacitance, maxCapacitance;
  PinType pinType;
  Cell* cell;

  MapOfTimingSense tSense;                      // timing sense
  MapOfTableSet cellRise, cellFall;             // cell delay
  MapOfTableSet riseTransition, fallTransition; // slew
  MapOfTableSet risePower, fallPower;           // power
};

struct Cell {
  std::string name, familyName;
  float area, cellLeakagePower;
  size_t driveStrength;
  std::unordered_map<std::string, CellPin*> outPins, inPins, internalPins,
      cellPins;
};

struct WireLoad {
  std::string name;
  float capacitance, resistance, slope;
  std::vector<size_t> fanout;
  std::vector<float> length;

  float wireResistance(size_t deg);
  float wireCapacitance(size_t deg);
};

struct CellLib {
  typedef std::unordered_map<std::string, Cell*> CellMap;

  std::string name;
  CellMap cells;
  std::unordered_map<std::string, CellMap> cellFamilies;
  std::unordered_map<std::string, LutTemplate*> lutTemplates;
  std::unordered_map<std::string, WireLoad*> wireLoads;

  WireLoad* defaultWireLoad;

  CellLib();
  ~CellLib();

  void read(std::string inName);
  void clear();
  void printDebug();
};

std::pair<float, std::string> extractMaxFromTableSet(CellPin::TableSet& tables,
                                                     std::vector<float>& param);

#endif // GALOIS_CELLLIB_H
