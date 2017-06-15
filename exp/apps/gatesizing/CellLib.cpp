#include "CellLib.h"
#include "FileReader.h"

#include <string>

static void readWireLoad(FileReader& fRd, CellLib *cellLib) {
  fRd.nextToken(); // get "("
  fRd.nextToken(); // get "\""

  WireLoad *wireLoad = new WireLoad;
  wireLoad->name = fRd.nextToken();
  cellLib->wireLoads.insert({wireLoad->name, wireLoad});
  std::cout << "wire " << wireLoad->name << std::endl;

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
      float length = std::stof(fRd.nextToken());
      wireLoad->fanoutLength.insert({fanout, length});
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
  std::cout << "lutTemplate: " << lutTemplate->name << std::endl;

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

static void readTableForCellPin(FileReader& fRd, CellLib *cellLib, Cell *cell, CellPin *cellPin) {
  fRd.nextToken(); // get "("
  fRd.nextToken(); // get ")"
  fRd.nextToken(); // get "{"

  std::string relatedPinName;
  TimingSense sense;
  for (std::string token = fRd.nextToken(); token != "}"; token = fRd.nextToken()) {
    if (token == "related_pin") {
      fRd.nextToken(); // get ":"
      fRd.nextToken(); // get "\""
      relatedPinName = fRd.nextToken();
      fRd.nextToken(); // get "\""
      fRd.nextToken(); // get ";"
    }

    else if (token == "timing_sense") {
      fRd.nextToken(); // get ":"
      token = fRd.nextToken();
      if (token == "positive_unate") {
        sense = TIMING_SENSE_POSITIVE_UNATE;
      }
      else if (token == "negative_unate") {
        sense = TIMING_SENSE_NEGATIVE_UNATE;
      }
      else {
        sense = TIMING_SENSE_NON_UNATE;
      }
    }

    else if (token == "cell_fall" || token == "cell_rise" 
             || token == "fall_transition" || token == "rise_transition"
             || token == "fall_power" || token == "rise_power") {

      auto& mapping = (token == "cell_fall") ? cellPin->cellFall : 
                      (token == "cell_rise") ? cellPin->cellRise :
                      (token == "fall_transition") ? cellPin->fallTransition : 
                      (token == "rise_transition") ? cellPin->riseTransition :
                      (token == "fall_power") ? cellPin->fallPower : cellPin->risePower;

      // skip the mapping if it exists
      if (mapping.count(relatedPinName)) {
        do {
          token = fRd.nextToken();
        } while (token != "}");
      }

      // read in the table
      else {
        fRd.nextToken(); // get "("

        LUT *lut = new LUT;
        lut->lutTemplate = cellLib->lutTemplates.at(fRd.nextToken());
        lut->tSense = sense;
        mapping.insert({relatedPinName, lut});

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

            fRd.nextToken(); // get ";"
          }
        } // end for token
      }
    }

    else {
      do {
        token = fRd.nextToken();
      } while (token != ";");
    }
  } // end for token
} // end readTableForCellPin

static void readCellPin(FileReader& fRd, CellLib *cellLib, Cell *cell) {
  fRd.nextToken(); // get "("

  CellPin *cellPin = new CellPin;
  cellPin->name = fRd.nextToken();

  fRd.nextToken(); // get ")"
  fRd.nextToken(); // get "{"

  for (std::string token = fRd.nextToken(); token != "}"; token = fRd.nextToken()) {
    if (token == "direction") {
      fRd.nextToken(); // get ":"
      token = fRd.nextToken();
      if (token == "input") {
        cell->inPins.insert({cellPin->name, cellPin});
      }
      else if (token == "output") {
        cell->outPins.insert({cellPin->name, cellPin});
      }
      else if (token == "internal") {
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
      readTableForCellPin(fRd, cellLib, cell, cellPin);
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
    std::cout << "token = " << token << std::endl;
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
      return;
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

CellLib::CellLib(std::string inName) {
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

static void printLUT(LUT *lut, std::string tableName, std::string pinName) {
  std::cout << "    " << tableName << "(" << pinName << ", " << lut->lutTemplate->name << ") {" << std::endl;

  for (size_t j = 0; j < lut->index.size(); j++) {
    std::cout << "      index_" << j << " (";
    for (size_t k = 0; k < lut->index[j].size(); k++) {
      std::cout << " " << lut->index[j][k];
    }
    std::cout << " )" << std::endl;
  }

  std::cout << "      value (" << std::endl;
  for (size_t j = 0; j < lut->value.size(); j++) {
    std::cout << "        ";
    for (size_t k = 0; k < lut->index[j].size(); k++) {
      std::cout << lut->index[j][k] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "      )" << std::endl;
  std::cout << "    }" << std::endl;
}

void CellLib::printCellLib() {
  std::cout << "library " << name << std::endl;

  for (auto item: wireLoads) {
    auto w = item.second;
    std::cout << "wire_load (" << w->name << ") {" << std::endl;
    std::cout << "  capacitance: " << w->capacitance << std::endl;
    std::cout << "  resistance: " << w->resistance << std::endl;
    std::cout << "  slope: " << w->slope << std::endl;
    for (auto i: w->fanoutLength) {
      std::cout << "  fanout_length(" << i.first << ", " << i.second << ")" << std::endl;
    }
    std::cout << "}" << std::endl;
  }

  std::cout << "default_wire_load: " << defaultWireLoad->name << std::endl;

  for (auto item: lutTemplates) {
    auto lutT = item.second;
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
  }
/*
  for (auto item: cells) {
    auto c = item.second;
    std::cout << "cell (" << c->name << ") {" << std::endl;
    std::cout << "  drive_strength: " << c->driveStrength << std::endl;
    std::cout << "  area: " << c->area << std::endl;
    std::cout << "  cell_leakage_power: " << c->cellLeakagePower << std::endl;

    for (auto item: c->inPins) {
      auto pin = item.second;
      std::cout << "  pin (" << pin->name << ") {" << std::endl;
      std::cout << "    direction: input" << std::endl;
      std::cout << "    capacitance: " << pin->capacitance << std::endl;
      std::cout << "  }" << std::endl;
    }

    for (auto item: c->internalPins) {
      auto pin = item.second;
      std::cout << "  pin (" << pin->name << ") {" << std::endl;
      std::cout << "    direction: internal" << std::endl;
      std::cout << "  }" << std::endl;
    }

    for (auto item: c->outPins) {
      auto pin = item.second;
      std::cout << "  pin (" << pin->name << ") {" << std::endl;
      std::cout << "    direction: output" << std::endl;

      for (auto i: pin->cellRise) {
        printLUT(i.second, "cell_rise", i.first);
      }
      for (auto i: pin->cellFall) {
        printLUT(i.second, "cell_fall", i.first);
      }
      for (auto i: pin->fallTransition) {
        printLUT(i.second, "fall_transition", i.first);
      }
      for (auto i: pin->riseTransition) {
        printLUT(i.second, "rise_transition", i.first);
      }
      for (auto i: pin->fallPower) {
        printLUT(i.second, "fall_power", i.first);
      }
      for (auto i: pin->risePower) {
        printLUT(i.second, "rise_power", i.first);
      }

      std::cout << "  }" << std::endl;
    }

    std::cout << "}" << std::endl;
  }
*/
}

CellLib::~CellLib() {
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
        delete j.second;
      }
      for (auto j: pin->cellFall) {
        delete j.second;
      }
      for (auto j: pin->riseTransition) {
        delete j.second;
      }
      for (auto j: pin->fallTransition) {
        delete j.second;
      }
      for (auto j: pin->risePower) {
        delete j.second;
      }
      for (auto j: pin->fallPower) {
        delete j.second;
      }
    }

    delete c;
  }
}
