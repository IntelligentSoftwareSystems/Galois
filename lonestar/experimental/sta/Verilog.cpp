#include "Verilog.h"

static std::string name0 = "1\'b0";
static std::string name1 = "1\'b1";

void VerilogParser::tokenizeFile(std::string inName) {
  std::vector<char> delimiters = {'(', ')', ',', ';', '\\', '[', ']', '=', '.'};
  std::vector<char> separators = {' ', '\t', '\n', '\r'};

  Tokenizer tokenizer(delimiters, separators);
  tokens = tokenizer.tokenize(inName);
  curToken = tokens.begin();
}

void VerilogParser::parseModule() {
  // module name (port1 [, port*]*);
  curToken += 1; // consume "module"
  module->name = *curToken;
  curToken += 2; // consume name and "("
  
  while (!isEndOfStatement()) {
    module->addPin(getVarName());
    curToken += 1; // consume port1 and ","/")"
  }
  curToken += 1; // consume ";"

  while (!isEndOfTokenStream() && "endmodule" != *curToken) {
    if ("input" == *curToken) {
      parseInPins();
    }
    else if ("output" == *curToken) {
      parseOutPins();
    }
    else if ("assign" == *curToken) {
      parseAssign();
    }
    else if ("wire" == *curToken) {
      parseWires();
    }
    else {
      parseSubModule();
    }
  }

  curToken += 1; // consume "endmodule"

  // connect wires and in/out pins
  for (auto& i: module->pins) {
    auto p = i.second;
    auto w = module->findWire(p->name);
    assert(w);
    p->wire = w;
    w->addPin(p);
  }
}

void VerilogParser::parseWires() {
  // wire wire1 [, wire*]*;
  while (!isEndOfStatement()) {
    curToken += 1; // consume "wire" or ","
    module->addWire(getVarName());
  }
  curToken += 1; // consume ";"
}

void VerilogParser::parseInPins() {
  // input pin1 [, pin*]*;
  while (!isEndOfStatement()) {
    curToken += 1; // consume "input" or ","
    auto pin = module->findPin(getVarName());
    assert(pin);
    module->addInPin(pin);
  }
  curToken += 1; // consume ";"
}

void VerilogParser::parseOutPins() {
  // output pin1 [, pin*]*;
  while (!isEndOfStatement()) {
    curToken += 1; // consume "output" or ","
    auto pin = module->findPin(getVarName());
    assert(pin);
    module->addOutPin(pin);
  }
  curToken += 1; // consume ";"
}

void VerilogParser::parseAssign() {
  // assign wireName = pinName;
  curToken += 1; // consume "assign"

  auto wire = module->findWire(getVarName());
  assert(wire);
  curToken += 1; // consume "="

  auto pin = module->findPin(getVarName());
  assert(pin);
  curToken += 1; // consume ";"
  
  wire->addPin(pin);
  pin->wire = wire;
}

void VerilogParser::parseSubModule() {
  // cellType name (.port1(wire1) [.port*(wire*)]*);
  Token cellType = *curToken;
  curToken += 1; // consume cellType
  Token name = getVarName();
  curToken += 1; // consume "("

  auto m = module->addSubModule(name, cellType);
  while (!isEndOfStatement()) {
    curToken += 1; // consume "."
    auto p = m->addPin(getVarName());
    curToken += 1; // consume "("
    auto w = module->findWire(getVarName());
    assert(w);
    curToken += 2; // consume ")" and ","/")"

    // connect the wire with the pin
    w->addPin(p);
    p->wire = w;
  }

  curToken += 1; // consume ";"
}

// tokens for the var name are consumed
Token VerilogParser::getVarName() {
  Token t = *curToken;
  auto prevToken = curToken;
  curToken += 1; // consume name or "\\"

  // variable in the from of "\\a[1]"
  if ("\\" == t) {
    for ( ; (!isEndOfTokenStream()) && ("]" != *prevToken); prevToken = curToken, curToken += 1) {
      t += *curToken;
    }
  }

  return t;
}

void VerilogParser::parse(std::string inName) {
  tokenizeFile(inName);
  while (!isEndOfTokenStream()) {
    // module ... endmodule
    if ("module" == *curToken) {
      parseModule();
    }
    else {
      std::cerr << inName << ": one of the top-level constructs is not a module. Aborts." << std::endl;
      exit(-1);
    }
  }
}

void VerilogPin::print(std::ostream& os) {
  if (module->inPins.count(this)) {
    os << "  input " << name << ";" << std::endl;
  }
  else if (module->outPins.count(this)) {
    os << "  output " << name << ";" << std::endl;
  }
  // skip non-interface pins
}

void VerilogWire::print(std::ostream& os) {
  // do not print wires for constants
  if (name != name0 && name != name1) {
    os << "  wire " << name << ";" << std::endl;
  }
}

void VerilogWire::addPin(VerilogPin* pin) {
  pins.insert(pin);
}

void VerilogModule::setup(bool isTopModule) {
  parentModule = this;

  if (isTopModule) {
    auto pin0 = addPin(name0);
    auto wire0 = addWire(name0);
    pin0->wire = wire0;
    wire0->addPin(pin0);

    auto pin1 = addPin(name1);
    auto wire1 = addWire(name1);
    pin1->wire = wire1;
    wire1->addPin(pin1);
  }
}

void VerilogModule::clear() {
  for (auto& i: subModules) {
    delete i.second;
  }

  for (auto& i: pins) {
    delete i.second;
  }

  for (auto& i: wires) {
    delete i.second;
  }
}

void VerilogModule::parse(std::string inName, bool toClear) {
  if (toClear) {
    clear();
    setup(true);
  }
  VerilogParser parser(this);
  parser.parse(inName);
}

void VerilogModule::print(std::ostream& os) {
  os << "module " << name << " (" << std::endl;

  size_t i = 0;
  for (auto& j: pins) {
    auto p = j.second; 
    os << "    " << p->name;
    if (i != (pins.size() - 1)) {
      os << ",";
    }
    os << std::endl;
    i++;
  };
  os << ");" << std::endl;

  for (auto& i: pins) {
    i.second->print(os);
  }

  for (auto& i: wires) {
    i.second->print(os);
  }

  for (auto op: outPins) {
    auto w = op->wire;
    for (auto p: w->pins) {
      if (p != op && (inPins.count(p) || p->name == name0 || p->name == name1)) {
        os << "  assign " << w->name << " = " << p->name << ";" << std::endl;
        break;
      }
    }
  }

  for (auto& j: subModules) {
    auto m = j.second;
    os << "  " << m->cellType << " " << m->name << "(";
    i = 0;
    for (auto& k: m->pins) {
      auto p = k.second;
      os << "." << p->name << "(" << p->wire->name << ")";
      if (i != m->pins.size()) {
        os << ", ";
      }
      else {
        os << ");";
      }
      i++;
    }
    os << std::endl;
  }

  os << "endmodule";
}

VerilogModule::VerilogModule(bool isTopModule) {
  setup(isTopModule);
}

VerilogModule::~VerilogModule() {
  clear();
}

VerilogModule* VerilogModule::addSubModule(std::string name, std::string cellType) {
  VerilogModule* m = new VerilogModule(false); // not the top-most module
  m->name = name;
  m->cellType = cellType;
  m->parentModule = this;
  subModules[name] = m;
  return m;
}

VerilogModule* VerilogModule::findSubModule(std::string name) {
  auto it = subModules.find(name);
  return (it == subModules.end()) ? nullptr : it->second;
}

VerilogPin* VerilogModule::addPin(std::string name) {
  VerilogPin* pin = new VerilogPin;
  pin->name = name;
  pin->module = this;
  pin->wire = nullptr;
  pins[name] = pin;
  return pin;
}

VerilogPin* VerilogModule::findPin(std::string name) {
  auto it = pins.find(name);
  return (it == pins.end()) ? nullptr : it->second;
}

VerilogWire* VerilogModule::addWire(std::string name) {
  VerilogWire* wire = new VerilogWire;
  wire->name = name;
  wire->module = this;
  wires[name] = wire;
  return wire;
}

VerilogWire* VerilogModule::findWire(std::string name) {
  auto it = wires.find(name);
  return (it == wires.end()) ? nullptr : it->second;
}
