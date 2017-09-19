#include "CellLib.h"
#include "FileReader.h"

#include <string>
#include <algorithm>
#include <utility>
#include <cassert>
#include <limits>

template<typename T>
static T linearInterpolate(T x1, T x2, T x3, T y1, T y3) {
  return y1 + (y3 - y1) * (x2 - x1) / (x3 - x1);
}

template<typename T>
static std::pair<size_t, size_t> findBound(T v, std::vector<T>& array) {
  auto upper = std::upper_bound(array.begin(), array.end(), v);
  if (upper == array.end()) {
    return std::make_pair(array.size()-1, array.size()-1);
  }
  if (upper == array.begin()) {
    return std::make_pair(0,0);
  }
  auto upperIndex = std::distance(array.begin(), upper);
  auto lowerIndex = upperIndex - 1;
  return std::make_pair(lowerIndex, upperIndex);
}

float WireLoad::wireResistance(size_t deg) {
  auto b = findBound(deg, fanout);
  float len;
  if (b.first != b.second) {
    len = linearInterpolate((float)fanout[b.first], (float)deg, (float)fanout[b.second], length[b.first], length[b.second]);
  }
  // out of lower bound
  else if (0 == b.second) {
    len = 0.0;
  }
  // out of upper bound
  else {
    len = length[b.second] + (deg - fanout[b.second]) * slope;
  }
  return resistance * len;
}

float WireLoad::wireCapacitance(size_t deg) {
  auto b = findBound(deg, fanout);
  float len;
  if (b.first != b.second) {
    len = linearInterpolate((float)fanout[b.first], (float)deg, (float)fanout[b.second], length[b.first], length[b.second]);
  }
  // out of lower bound
  else if (0 == b.second) {
    len = 0.0;
  }
  // out of upper bound
  else {
    len = length[b.second] + (deg - fanout[b.second]) * slope;
  }
  return capacitance * len;
}

float LUT::lookup(std::vector<float>& param) {
  // dimensions should match
  auto paramDim = param.size();
  assert(paramDim == index.size());

  auto b0 = findBound(param[0], index[0]);
  if (1 == paramDim) {
    return linearInterpolate(index[0][b0.first], param[0], index[0][b0.second], value[0][b0.first], value[0][b0.second]);
  }
  else {
    auto b1 = findBound(param[1], index[1]);
    auto y1 = linearInterpolate(index[1][b1.first], param[1], index[1][b1.second], value[b0.first][b1.first], value[b0.first][b1.second]);
    auto y3 = linearInterpolate(index[1][b1.first], param[1], index[1][b1.second], value[b0.second][b1.first], value[b0.second][b1.second]);
    return linearInterpolate(index[0][b0.first], param[0], index[0][b0.second], y1, y3);
  }
}

static void readWireLoad(FileReader& fRd, CellLib *cellLib) {
  fRd.nextToken(); // get "("
  fRd.nextToken(); // get "\""

  WireLoad *wireLoad = new WireLoad;
  wireLoad->name = fRd.nextToken();
  cellLib->wireLoads.insert({wireLoad->name, wireLoad});

  fRd.nextToken(); // get "\""
  fRd.nextToken(); // get ")"
  fRd.nextToken(); // get "{"

  for (std::string token = fRd.nextToken(); token != "}"; token = fRd.nextToken()) {
    if (token == "capacitance") {
      fRd.nextToken(); // get ":"
      wireLoad->capacitance = std::stof(fRd.nextToken());
    }
    else if (token == "resistance") {
      fRd.nextToken(); // get ":"
      wireLoad->resistance = std::stof(fRd.nextToken());    
    }
    else if (token == "slope") {
      fRd.nextToken(); // get ":"
      wireLoad->slope = std::stof(fRd.nextToken());
    }
    else if (token == "fanout_length") {
      fRd.nextToken(); // get "("
      size_t fanout = std::stoul(fRd.nextToken());
      wireLoad->fanout.push_back(fanout);
      float length = std::stof(fRd.nextToken());
      wireLoad->length.push_back(length);
      fRd.nextToken(); // get ")"
      fRd.nextToken(); // get ";"
    }
  } // end for token
} // end readWireLoad

static void readLutTemplate(FileReader& fRd, CellLib *cellLib) {
  fRd.nextToken(); // get "("

  LutTemplate *lutTemplate = new LutTemplate;
  lutTemplate->name = fRd.nextToken();
  cellLib->lutTemplates.insert({lutTemplate->name, lutTemplate});

  fRd.nextToken(); // get ")"
  fRd.nextToken(); // get "{"

  for (std::string token = fRd.nextToken(); token != "}"; token = fRd.nextToken()) {
    if (token == "variable_1" || token == "variable_2") {
      fRd.nextToken(); // get ":"
      token = fRd.nextToken();
      if (token == "input_transition_time" || token == "input_net_transition") {
        lutTemplate->var.push_back(LUT_VAR_INPUT_NET_TRANSITION);
      } 
      else if (token == "total_output_net_capacitance") {
        lutTemplate->var.push_back(LUT_VAR_TOTAL_OUTPUT_NET_CAPACITANCE);
      }
      else if (token == "constrained_pin_transition") {
        lutTemplate->var.push_back(LUT_VAR_CONSTRAINED_PIN_TRANSITION);
      }
      else if (token == "related_pin_transition") {
        lutTemplate->var.push_back(LUT_VAR_RELATED_PIN_TRANSITION);
      }
      else {
        lutTemplate->var.push_back(LUT_VAR_UNDEFINED);
      }
      fRd.nextToken(); // get ";"
    }
    else if (token == "index_1" || token == "index_2") {
      fRd.nextToken(); // get "("
      fRd.nextToken(); // get "\""
      size_t dimension = 0;
      for (token = fRd.nextToken(); token != "\""; token = fRd.nextToken()) {
        dimension++;
      }
      lutTemplate->dim.push_back(dimension);
      fRd.nextToken(); // get ")"
      fRd.nextToken(); // get ";" 
    }
  } // end for token
} // end readLutTemplate

static void printTimingSense(TimingSense t) {
  std::cout << ((t == TIMING_SENSE_POSITIVE_UNATE) ? "positive_unate" :
                (t == TIMING_SENSE_NEGATIVE_UNATE) ? "negative_unate" :
                (t == TIMING_SENSE_NON_UNATE) ? "non-unate" : "undefined"
               );
}

static void readLutForCellPin(FileReader& fRd, CellLib *cellLib, Cell *cell, CellPin *cellPin) {
  fRd.nextToken(); // get "("
  fRd.nextToken(); // get ")"
  fRd.nextToken(); // get "{"

  std::string relatedPinName, whenStr;
  for (std::string token = fRd.nextToken(); token != "}"; token = fRd.nextToken()) {
    if (token == "related_pin") {
      fRd.nextToken(); // get ":"
      fRd.nextToken(); // get "\""
      relatedPinName = fRd.nextToken();
      fRd.nextToken(); // get "\""
      fRd.nextToken(); // get ";"
    }

    else if (token == "when") {
      fRd.nextToken(); // get ":"
      fRd.nextToken(); // get "\""
      whenStr = fRd.nextToken();
      for (token = fRd.nextToken(); token != "\""; token = fRd.nextToken()) {
        whenStr += " " + token;
      }
      fRd.nextToken(); // get ";"
    }

    else if (token == "timing_sense") {
      fRd.nextToken(); // get ":"

      token = fRd.nextToken();
      auto& mapping = cellPin->tSense;
      if (token == "positive_unate") {
        mapping.insert({{relatedPinName, whenStr}, TIMING_SENSE_POSITIVE_UNATE});
      }
      else if (token == "negative_unate") {
        mapping.insert({{relatedPinName, whenStr}, TIMING_SENSE_NEGATIVE_UNATE});
      }
      else {
        // not handling unateness other than positive/negative unate
        mapping.insert({{relatedPinName, whenStr}, TIMING_SENSE_NON_UNATE});
      }
      fRd.nextToken(); // get ";"
    }

    else if (token == "cell_fall" || token == "cell_rise" 
             || token == "fall_transition" || token == "rise_transition"
             || token == "fall_power" || token == "rise_power") {

      auto& mapping = (token == "cell_fall") ? cellPin->cellFall : 
                      (token == "cell_rise") ? cellPin->cellRise :
                      (token == "fall_transition") ? cellPin->fallTransition : 
                      (token == "rise_transition") ? cellPin->riseTransition :
                      (token == "fall_power") ? cellPin->fallPower : cellPin->risePower;

      auto tSense = cellPin->tSense[{relatedPinName, whenStr}];
      auto& tables = mapping[{relatedPinName, tSense}];

      // read in the new table
      fRd.nextToken(); // get "("

      LUT *lut = new LUT;
      lut->lutTemplate = cellLib->lutTemplates.at(fRd.nextToken());
      tables.insert({whenStr, lut});

      fRd.nextToken(); // get ")"
      fRd.nextToken(); // get "{"

      for (token = fRd.nextToken(); token != "}"; token = fRd.nextToken()) {
        if (token == "index_1" || token == "index_2") {
          fRd.nextToken(); // get "("
          fRd.nextToken(); // get "\""

          std::vector<float> v;
          for (token = fRd.nextToken(); token != "\""; token = fRd.nextToken()) {
            v.push_back(stof(token));
          }
          lut->index.push_back(v);

          fRd.nextToken(); // get ")"
          fRd.nextToken(); // get ";"            
        }
        else if (token == "values") {
          fRd.nextToken(); // get "("

          std::vector<float> v;
          for (token = fRd.nextToken(); token != ")"; token = fRd.nextToken()) {
           if (token == "\\") {
              lut->value.push_back(v);
              v.clear();
            } else if (token == "\"") {
              // skip
            } else {
              v.push_back(stof(token));
            }
          }
          lut->value.push_back(v);
          v.clear();
          fRd.nextToken(); // get ";"
        }
      } // end for token
    }

    else if (token == "fall_constraint" || token == "rise_constraint") {
      do {
        token = fRd.nextToken();
      } while (token != "}");
    }

    else {
      do {
        token = fRd.nextToken();
      } while (token != ";");
    }
  } // end for token
} // end readLutForCellPin

static void readCellPin(FileReader& fRd, CellLib *cellLib, Cell *cell) {
  fRd.nextToken(); // get "("

  CellPin *cellPin = new CellPin;
  cellPin->name = fRd.nextToken();
  cellPin->pinType = PIN_UNDEFINED;
  cellPin->cell = cell;
  cell->cellPins.insert({cellPin->name, cellPin});

  fRd.nextToken(); // get ")"
  fRd.nextToken(); // get "{"

  for (std::string token = fRd.nextToken(); token != "}"; token = fRd.nextToken()) {
    if (token == "direction") {
      fRd.nextToken(); // get ":"
      token = fRd.nextToken();
      if (token == "input") {
        cellPin->pinType = PIN_INPUT;
        cell->inPins.insert({cellPin->name, cellPin});
      }
      else if (token == "output") {
        cellPin->pinType = PIN_OUTPUT;
        cell->outPins.insert({cellPin->name, cellPin});
      }
      else if (token == "internal") {
        cellPin->pinType = PIN_INTERNAL;
        cell->internalPins.insert({cellPin->name, cellPin});
      }
      fRd.nextToken(); // get ";"
    }

    else if (token == "capacitance") {
      fRd.nextToken(); // get ":"
      cellPin->capacitance = std::stof(fRd.nextToken());
      fRd.nextToken(); // get ";"
    }

    else if (token == "timing" || token == "internal_power") {
      readLutForCellPin(fRd, cellLib, cell, cellPin);
    }

    else {
      do {
        token = fRd.nextToken();
      } while (token != ";");
    }
  } // end for token
} // end readCellPin

static void readCell(FileReader& fRd, CellLib *cellLib) {
  fRd.nextToken(); // get "("

  Cell *cell = new Cell;
  cell->name = fRd.nextToken();
  cellLib->cells.insert({cell->name, cell});
  cell->familyName = cell->name.substr(0, cell->name.find("_"));
  cellLib->cellFamilies[cell->familyName].insert({cell->name, cell});

  fRd.nextToken(); // get ")"
  fRd.nextToken(); // get "{"

  for (std::string token = fRd.nextToken(); token != "}"; token = fRd.nextToken()) {
    if (token == "drive_strength") {
      fRd.nextToken(); // get ":"
      cell->driveStrength = std::stoul(fRd.nextToken());
      fRd.nextToken(); // get ";"
    }

    else if (token == "area") {
      fRd.nextToken(); // get ":"
      cell->area = std::stof(fRd.nextToken());
      fRd.nextToken(); // get ";"
    }

    else if (token == "cell_leakage_power") {
      fRd.nextToken(); // get ":"
      cell->cellLeakagePower = std::stof(fRd.nextToken());
      fRd.nextToken(); // get ";"
    }

    else if (token == "pin") {
      readCellPin(fRd, cellLib, cell);
    }

    else if (token == "pg_pin" || token == "leakage_power" 
             || token == "statetable" || token == "ff" || token == "latch") {
      do {
        token = fRd.nextToken();
      } while (token != "}");
    }

    else if (token == "dont_touch" || token == "dont_use" || token == "clock_gating_integrated_cell") {
      do {
        token = fRd.nextToken();
      } while (token != ";");    
    }
  } // end for token
} // end readCell

static void readCellLibBody(FileReader& fRd, CellLib *cellLib) {
  // parse until hits the end "}" of library
  for (std::string token = fRd.nextToken(); token != "}"; token = fRd.nextToken()) {
    if (token == "wire_load") {
      readWireLoad(fRd, cellLib);
    }

    else if (token == "default_wire_load") {
      fRd.nextToken(); // get ":"
      fRd.nextToken(); // get "\""
      cellLib->defaultWireLoad = cellLib->wireLoads.at(fRd.nextToken());
      fRd.nextToken(); // get "\""
      fRd.nextToken(); // get ";" 
    }

    else if (token == "power_lut_template" || token == "lu_table_template") {
      readLutTemplate(fRd, cellLib);
    }

    else if (token == "cell") {
      readCell(fRd, cellLib);
    }

    else if (token == "operating_conditions") {
      do {
        token = fRd.nextToken();
      } while (token != "}");
    }

    else {
      do {
        token = fRd.nextToken();
      } while (token != ";");
    }
  } // end for token
} // end readCellLibBody

static void readCellLib(FileReader& fRd, CellLib *cellLib) {
  for (std::string token = fRd.nextToken(); token != ""; token = fRd.nextToken()) {
    // library (libraryName) { ... }
    if (token == "library") {
      fRd.nextToken(); // get "("
      cellLib->name = fRd.nextToken();
      fRd.nextToken(); // get ")"
      fRd.nextToken(); // get "{"
      readCellLibBody(fRd, cellLib);
    }
  }
}

void CellLib::read(std::string inName) {
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

  // put scalar lut_template in
  LutTemplate *scalar = new LutTemplate;
  scalar->name = "scalar";
  scalar->var.push_back(LUT_VAR_UNDEFINED);
  scalar->dim.push_back(1);
  lutTemplates.insert({scalar->name, scalar});

  readCellLib(fRd, this);
}

static void printWireLoad(WireLoad *w) {
  std::cout << "wire_load (" << w->name << ") {" << std::endl;
  std::cout << "  capacitance: " << w->capacitance << std::endl;
  std::cout << "  resistance: " << w->resistance << std::endl;
  std::cout << "  slope: " << w->slope << std::endl;
  for (size_t i = 0; i < w->fanout.size(); i++) {
    std::cout << "  fanout_length(" << w->fanout[i] << ", " << w->length[i] << ")" << std::endl;
  }
  std::cout << "}" << std::endl;
}

static void printLutTemplate(LutTemplate *lutT) {
  std::cout << "lu_table_template (" << lutT->name << ") {" << std::endl;
  for (size_t i = 0; i < lutT->var.size(); i++) {
    std::cout << "  variable_" << i << ": ";
    std::cout << ((lutT->var[i] == LUT_VAR_INPUT_NET_TRANSITION) ? "input_net_transition" :
                  (lutT->var[i] == LUT_VAR_TOTAL_OUTPUT_NET_CAPACITANCE) ? "total_output_capacitance" : 
                  (lutT->var[i] == LUT_VAR_CONSTRAINED_PIN_TRANSITION) ? "constrained_pin_transition" : 
                  (lutT->var[i] == LUT_VAR_RELATED_PIN_TRANSITION) ? "related_pin_transition" : "undefined");
    std::cout << std::endl;
  }

  for (size_t i = 0; i < lutT->dim.size(); i++) {
    std::cout << "  index_" << i << ": " << lutT->dim[i] << std::endl;
  }
  std::cout << "}" << std::endl;
}

static void printLUT(LUT *lut, std::string tableName, CellPin::TableSetKey& key, std::string whenStr) {
  std::string pinName = key.first;
  TimingSense t = key.second;

  std::cout << "      " << tableName << "(" << pinName << ", ";
  printTimingSense(t);
  std::cout << " when " << whenStr << ", " << lut->lutTemplate->name << ") {" << std::endl;

  for (size_t j = 0; j < lut->index.size(); j++) {
    std::cout << "        index_" << j << " (";
    for (size_t k = 0; k < lut->index[j].size(); k++) {
      std::cout << " " << lut->index[j][k];
    }
    std::cout << " )" << std::endl;
  }

  std::cout << "        value (" << std::endl;
  for (size_t j = 0; j < lut->value.size(); j++) {
    std::cout << "          ";
    for (size_t k = 0; k < lut->value[j].size(); k++) {
      std::cout << lut->value[j][k] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "        )" << std::endl;
  std::cout << "      }" << std::endl;
}

static void printTableSet(CellPin::TableSet& tables, std::string tableName, CellPin::TableSetKey key) {
  for (auto& i: tables) {
    printLUT(i.second, tableName, key, i.first);
  }
}

static void printCell(Cell *c) {
  std::cout << "  cell (" << c->name << " in " << c->familyName << ") {" << std::endl;
  std::cout << "    drive_strength: " << c->driveStrength << std::endl;
  std::cout << "    area: " << c->area << std::endl;
  std::cout << "    cell_leakage_power: " << c->cellLeakagePower << std::endl;

  for (auto item: c->inPins) {
    auto pin = item.second;
    std::cout << "    pin (" << pin->name << ") {" << std::endl;
    std::cout << "      direction: input" << std::endl;
    std::cout << "      capacitance: " << pin->capacitance << std::endl;
    std::cout << "    }" << std::endl;
  }

  for (auto item: c->internalPins) {
    auto pin = item.second;
    std::cout << "    pin (" << pin->name << ") {" << std::endl;
    std::cout << "      direction: internal" << std::endl;
    std::cout << "    }" << std::endl;
  }

  for (auto item: c->outPins) {
    auto pin = item.second;
    std::cout << "    pin (" << pin->name << ") {" << std::endl;
    std::cout << "      direction: output" << std::endl;

    for (auto i: pin->tSense) {
      auto inPinName = i.first.first;
      auto inPinWhen = i.first.second;
      auto inPinSense = i.second;
      std::cout << "      timing sense for input pin " << inPinName << ": ";
      printTimingSense(inPinSense);
      std::cout << " when " << ((inPinWhen == "") ? "(null)" : inPinWhen) << std::endl;
    }

    for (auto i: pin->cellRise) {
      printTableSet(i.second, "cell_rise", i.first);
    }
    for (auto i: pin->cellFall) {
      printTableSet(i.second, "cell_fall", i.first);
    }
    for (auto i: pin->fallTransition) {
      printTableSet(i.second, "fall_transition", i.first);
    }
    for (auto i: pin->riseTransition) {
      printTableSet(i.second, "rise_transition", i.first);
    }
    for (auto i: pin->fallPower) {
      printTableSet(i.second, "fall_power", i.first);
    }
    for (auto i: pin->risePower) {
      printTableSet(i.second, "rise_power", i.first);
    }
    std::cout << "    }" << std::endl;
  }

  std::cout << "  }" << std::endl;
}

void CellLib::printDebug() {
  std::cout << "library " << name << std::endl;

  for (auto item: wireLoads) {
    printWireLoad(item.second);
  }

  std::cout << "default_wire_load: " << defaultWireLoad->name << std::endl;

  for (auto item: lutTemplates) {
    printLutTemplate(item.second);
  }

  for (auto item: cellFamilies) {
    std::cout << "Cell Family " << item.first << " {" << std::endl;
    auto& cf = item.second;
    for (auto i: cf) {
      printCell(i.second);
    }
    std::cout << "}" << std::endl;
  }
}

void CellLib::clear() {
  for (auto item: wireLoads) {
    delete item.second;
  }

  for (auto item: lutTemplates) {
    delete item.second;
  }

  for (auto item: cells) {
    auto c = item.second;

    for (auto i: c->outPins) {
      auto pin = i.second;

      // free LUTs
      for (auto j: pin->cellRise) {
        for (auto k: j.second) {
          delete k.second;
        }
      }
      for (auto j: pin->cellFall) {
        for (auto k: j.second) {
          delete k.second;
        }
      }
      for (auto j: pin->riseTransition) {
        for (auto k: j.second) {
          delete k.second;
        }
      }
      for (auto j: pin->fallTransition) {
        for (auto k: j.second) {
          delete k.second;
        }
      }
      for (auto j: pin->risePower) {
        for (auto k: j.second) {
          delete k.second;
        }
      }
      for (auto j: pin->fallPower) {
        for (auto k: j.second) {
          delete k.second;
        }
      }
    }

    delete c;
  }
}

CellLib::CellLib() {
}

CellLib::~CellLib() {
  clear();
}

float extractMaxFromTableSet(CellPin::TableSet& tables, std::vector<float>& param) {
  float r = -std::numeric_limits<float>::infinity();
  for (auto& i: tables) {
    float tmp = i.second->lookup(param);
    if (tmp > r) {
      r = tmp;
    }
  }
  return r;
}
