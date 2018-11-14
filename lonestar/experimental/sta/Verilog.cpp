#include "Verilog.h"

static std::string name0 = "1\'b0";
static std::string name1 = "1\'b1";

static bool isNameForVdd(std::string name) {
  return (name == name0 || name == "1\'h0");
}

static bool isNameForGnd(std::string name) {
  return (name == name1 || name == "1\'h1");
}

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
  curModule = design.addModule(*curToken);
  curToken += 2; // consume name and "("
  
  while (!isEndOfStatement()) {
    curModule->addPin(getVarName());
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
      parseGate();
    }
  }

  curToken += 1; // consume "endmodule"

  // connect wires and in/out pins
  for (auto& i: curModule->pins) {
    auto p = i.second;
    auto w = curModule->findWire(p->name);
    assert(w);
    w->addPin(p);

    // no assign statement for this pin
    if (!(p->wire)) {
      p->wire = w;
    }
  }
}

void VerilogParser::parseWires() {
  // wire wire1 [, wire*]*;
  while (!isEndOfStatement()) {
    curToken += 1; // consume "wire" or ","
    curModule->addWire(getVarName());
  }
  curToken += 1; // consume ";"
}

void VerilogParser::parseInPins() {
  // input pin1 [, pin*]*;
  while (!isEndOfStatement()) {
    curToken += 1; // consume "input" or ","
    auto pin = curModule->findPin(getVarName());
    assert(pin);
    curModule->addInPin(pin);
  }
  curToken += 1; // consume ";"
}

void VerilogParser::parseOutPins() {
  // output pin1 [, pin*]*;
  while (!isEndOfStatement()) {
    curToken += 1; // consume "output" or ","
    auto pin = curModule->findPin(getVarName());
    assert(pin);
    curModule->addOutPin(pin);
  }
  curToken += 1; // consume ";"
}

void VerilogParser::parseAssign() {
  // assign pinName = wireName;
  curToken += 1; // consume "assign"

  auto pin = curModule->findPin(getVarName());
  assert(pin);
  curToken += 1; // consume "="

  auto wire = curModule->findWire(getVarName());
  assert(wire);
  curToken += 1; // consume ";"
  
  pin->wire = wire;
  wire->addPin(pin);
}

void VerilogParser::parseGate() {
  // cellType name (.port1(wire1) [.port*(wire*)]*);
  Token cellType = *curToken;
  curToken += 1; // consume cellType
  Token name = getVarName();
  curToken += 1; // consume "("

  auto g = curModule->addGate(name, cellType);
  while (!isEndOfStatement()) {
    curToken += 1; // consume "."
    auto p = g->addPin(getVarName());
    curToken += 1; // consume "("
    auto w = curModule->findWire(getVarName());
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
  if (!isNameForGnd(name) && !isNameForVdd(name)) {
    os << "  wire " << name << ";" << std::endl;
  }
}

void VerilogGate::print(std::ostream& os) {
  os << "  " << cellType << " " << name << "(";
  size_t i = 1;
  for (auto& j: pins) {
    auto p = j.second;
    os << "." << p->name << "(" << p->wire->name << ")";
    if (i != pins.size()) {
      os << ", ";
    }
    else {
      os << ");";
    }
    i++;
  }
  os << std::endl;
}

VerilogGate::~VerilogGate() {
  for (auto& i: pins) {
    delete i.second;
  }
}

VerilogModule::VerilogModule() {
  auto pin0 = addPin(name0);
  auto wire0 = addWire(name0);
  pin0->wire = wire0;
  wire0->addPin(pin0);

  auto pin1 = addPin(name1);
  auto wire1 = addWire(name1);
  pin1->wire = wire1;
  wire1->addPin(pin1);
}

VerilogModule::~VerilogModule() {
  for (auto& i: gates) {
    delete i.second;
  }

  for (auto& i: pins) {
    delete i.second;
  }

  for (auto& i: wires) {
    delete i.second;
  }
}

void VerilogModule::print(std::ostream& os) {
  os << "module " << name << " (" << std::endl;

  size_t i = 1;
  size_t numRealPins = pins.size() - pins.count(name0) - pins.count(name1);
  for (auto& j: pins) {
    auto p = j.second; 
    if (p->name != name0 && p->name != name1) {
      os << "    " << p->name;
      if (i != numRealPins) {
        os << ",";
      }
      os << std::endl;
      i++;
    }
  }
  os << ");" << std::endl;

  for (auto& i: pins) {
    i.second->print(os);
  }

  for (auto& i: wires) {
    i.second->print(os);
  }

  for (auto& i: outPins) {
    auto& pinName = i->name;
    auto& wireName = i->wire->name;
    if (pinName != wireName) {
      os << "  assign " << pinName << " = " << wireName << ";" << std::endl;
    }
  }

  for (auto& i: gates) {
    i.second->print(os);
  }

  os << "endmodule" << std::endl;
}

bool VerilogModule::isHierarchical() {
  return !succ.empty();
}

VerilogWire* VerilogModule::findWire(std::string name) {
  name = isNameForGnd(name) ? name0 :
         isNameForVdd(name) ? name1 : name;
  auto it = wires.find(name);
  return (it == wires.end()) ? nullptr : it->second;
}

void VerilogDesign::clear() {
  clearDependency();
  for (auto& i: modules) {
    delete i.second;
  }
  modules.clear();
}

void VerilogDesign::clearDependency() {
  for (auto& i: modules) {
    auto m = i.second;
    m->pred.clear();
    m->succ.clear();
  }
  roots.clear();
}

void VerilogDesign::buildDependency() {
  clearDependency();

  for (auto& i: modules) {
    auto m = i.second;
    for (auto& j: m->gates) {
      auto g = j.second;
      auto used = findModule(g->cellType);
      if (used) {
        m->succ.insert(used);
        used->pred.insert(m);
      }
    }
  }

  for (auto& i: modules) {
    auto m = i.second;
    if (m->pred.empty()) {
      roots.insert(m);
    }
  }
}

bool VerilogDesign::isHierarchical() {
  for (auto m: roots) {
    if (m->isHierarchical()) {
      return true;
    }
  }
  return false;
}

void VerilogDesign::print(std::ostream& os) {
  for (auto& i: modules) {
    i.second->print(os);
  }
}

void VerilogDesign::parse(std::string inName, bool toClear) {
  if (toClear) {
    clear();
  }
  VerilogParser parser(*this);
  parser.parse(inName);
}
