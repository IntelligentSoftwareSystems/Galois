#include "Verilog.h"
#include "FileReader.h"

void VerilogModule::clear() {
  for (auto item: inputs) {
     delete item.second;
  }

  for (auto item: outputs) {
    delete item.second;
  }

  for (auto item: gates) {
    delete item.second;
  }

  for (auto item: wires) {
    delete item.second;
  }
}

VerilogModule::~VerilogModule() {
  clear();
}

void VerilogModule::read(std::string inName, CellLib *lib) {
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
  cellLib = lib;

  for (std::string token = fRd.nextToken(); token != ""; token = fRd.nextToken()) {
    // module moduleName(port1, port2, ...);
    if (token == "module") {
      name = fRd.nextToken();
      do {
        token = fRd.nextToken();
      } while (token != ";");
    }

    // input/output port1, port2, ..., portN;
    else if (token == "input" || token == "output") {
      auto& primary = (token == "input") ? inputs : outputs;
      for (token = fRd.nextToken(); token != ";"; token = fRd.nextToken()) {
        VerilogPin *pin = new VerilogPin;
        pin->name = token;
        pin->gate = nullptr;
        pin->wire = nullptr;
        primary.insert({token, pin});
      }
    }

    // wire wire1, wire2, ..., wireN;
    else if (token == "wire") {
      for (token = fRd.nextToken(); token != ";"; token = fRd.nextToken()) {
        VerilogWire *wire = new VerilogWire;
        wire->name = token;
        wire->root = nullptr;
        wire->wireLoad = cellLib->defaultWireLoad;
        wires.insert({token, wire});
      }
    }

    else if (token == "endmodule") {
      break;
    }

    // logic gates: gateType gateName ( .port1 (wire1), .port2 (wire2), ... .portN (wireN) );
    else {
      VerilogGate *gate = new VerilogGate;
      gate->cell = cellLib->cells.at(token);
      gate->name = fRd.nextToken();
      gates.insert({gate->name, gate});
      fRd.nextToken(); // get "("

      // get pins and wire connections
      for (token = fRd.nextToken(); token != ")"; token = fRd.nextToken()) {
        if (token[0] != '.') {
          std::cerr << "Error: expecting .pinName(wrieName)" << std::endl;
          std::abort();
        }
        // .pinName (wireName)
        VerilogPin *pin = new VerilogPin;
        pin->name = token.substr(1);
        pin->gate = gate;

        fRd.nextToken(); // get "("
        pin->wire = wires.at(fRd.nextToken());
        fRd.nextToken(); // get ")"

        auto cellPin = gate->cell->cellPins.at(pin->name);
        if (cellPin->pinType == PIN_OUTPUT) {
          pin->wire->root = pin;
          gate->outPins.insert(pin);
        }
        else if (cellPin->pinType == PIN_INPUT) {
          pin->wire->leaves.insert(pin);
          gate->inPins.insert(pin);
        }
      }
      fRd.nextToken(); // get ";"
    }
  } // end for token

  // connect input to input wire
  for (auto item: inputs) {
    auto i = item.second;
    auto wire = wires.at(i->name);
    i->wire = wire;
    wire->root = i;
  }
  // connect output to output wire
  for (auto item: outputs) {
    auto i = item.second;
    auto wire = wires.at(i->name);
    i->wire = wire;
    wire->leaves.insert(i);
  }
}

VerilogModule::VerilogModule() {
}

void VerilogModule::printVerilogModuleDebug() {
  std::cout << "module " << name << std::endl;
  for (auto item: inputs) {
    auto i = item.second;
    std::cout << "input " << i->name << std::endl;
  }

  for (auto item: outputs) {
    auto o = item.second;
    std::cout << "output " << o->name << std::endl;
  }

  for (auto item: wires) {
    auto w = item.second;
    std::cout << "wire " << w->name << ": from ";
    // input/output wires don't have gates
    if (w->root->gate) {
      std::cout << w->root->gate->name << ".";
    }
    std::cout << w->root->name << " to ";

    for (auto p: w->leaves) {
      // input/output wires don't have gates
      if (p->gate) {
        std::cout << p->gate->name << ".";
      }
      std::cout << p->name << " ";
    }
    std::cout << std::endl;
  }

  for (auto item: gates) {
    auto g = item.second;
    std::cout << "gate: " << g->cell->name << " " << g->name << "(";
    for (auto p: g->inPins) {
      std::cout << "." << p->name << " (" << p->wire->name << ") ";
    }
    for (auto p: g->outPins) {
      std::cout << "." << p->name << " (" << p->wire->name << ") ";
    }
    std::cout << ");" << std::endl;
  }
}

static void writeVerilogIOs(std::ofstream& of, std::string portTypeName, std::unordered_map<std::string, VerilogPin *>& ports) {
  // input/output port1, port2, ...;
  of << "  " << portTypeName << " ";
  size_t i = 0, num = ports.size();
  for (auto item: ports) {
    of << item.second->name;
    num--;
    i++;
    if (num) {
      of << ", ";
      if (0 == i % 10) {
        of << "\n      ";
      }
    }
    else {
      of << ";" << std::endl;
    }
  }
}

static void writeVerilogWires(std::ofstream& of, std::unordered_map<std::string, VerilogWire *>& wires) {
  size_t i = 0, num = wires.size();

  // wire wire1, wire2, ...;
  of << "  wire ";
  for (auto item: wires) {
    of << item.second->name;
    num--;
    i++;
    if (num) {
      of << ((0 == i % 10) ? ";\n  wire " : ", ");
    }
    else {
      of << ";" << std::endl;
    }
  }
}

void VerilogModule::writeVerilogModule(std::string outName) {
  std::ofstream of(outName);
  if (!of.is_open()) {
    std::cerr << "Cannot open " << outName << " to write." << std::endl;
    return;
  }

  size_t i = 0, num;

  // module moduleName (port1, port2, ...);
  of << "module " << name << "(";
  num = inputs.size() + outputs.size();
  for (auto item: inputs) {
    of << item.second->name;
    num--;
    i++;
    if (num) {
      of << ", ";
      if (0 == i % 10) {
        of << "\n    ";
      }
    }
    else {
      of << ");" << std::endl;
    }
  }
  for (auto item: outputs) {
    of << item.second->name;
    num--;
    i++;
    if (num) {
      of << ", ";
      if (0 == i % 10) {
        of << "\n    ";
      }
    }
    else {
      of << ");" << std::endl;
    }
  }

  writeVerilogIOs(of, "input", inputs);
  writeVerilogIOs(of, "output", outputs);

  writeVerilogWires(of, wires);

  for (auto item: gates) {
    auto g = item.second;
    of << "  " << g->cell->name << " " << g->name << "(";
    num = g->cell->cellPins.size();
    for (auto p: g->inPins) {
      of << "." << p->name << " (" << p->wire->name << ")";
      num--;
      of << ((num) ? ", " : ");");
    }
    for (auto p: g->outPins) {
      of << "." << p->name << " (" << p->wire->name << ")";
      num--;
      of << ((num) ? ", " : ");");
    }
    of << std::endl;
  }

  of << "endmodule" << std::endl;
}

