#include "CellLib.h"
#include "FileReader.h"

#include <string>

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
    }
    else if (token == "index_1" || token == "index_2") {
      fRd.nextToken(); // get "("
      fRd.nextToken(); // get "\""
      size_t dimension = 0;
      for (token = fRd.nextToken(); token != "\""; token = fRd.nextToken()) {
        dimension++;
      }
      lutTemplate->dim.push_back(dimension);
      fRd.nextToken(); // get "\""
      fRd.nextToken(); // get ")"
      fRd.nextToken(); // get ";" 
    }
  } // end for token
} // end readLutTemplate

static void readTimingTableForCellPin(FileReader& fRd, CellLib *cellLib, Cell *cell, CellPin *cellPin) {
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

    else if (token == "cell_rise") {
      
    }

    else {
      do {
        token = fRd.nextToken();
      } while (token != ";");
    }
  } // end for token
} // end readTimingTableForCellPin

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

    else if (token == "timing") {
      readTimingTableForCellPin(fRd, cellLib, cell, cellPin);
    }

    else if (token == "internal_power") {
//      readPowerTableForCellPin(fRd, cellLib, cell, cellPin);
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
  std::cout << "CellLib " << inName << std::endl;
  readCellLib(fRd, this);
  std::cout << "End CellLib" << inName << std::endl;
}

CellLib::~CellLib() {
}
