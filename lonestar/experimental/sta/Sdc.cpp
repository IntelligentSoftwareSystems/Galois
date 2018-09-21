#include "Sdc.h"

#include <cstdlib>
#include <algorithm>

using PinSet = std::unordered_set<VerilogPin*>;

void SDCParser::collectPort(PinSet& ports, Token name) {
  auto pin = sdc.m.findPin(name);
  if (pin) {
    ports.insert(pin);
  }
  else {
    std::cout << "Warning: cannot find port " << name << " in sdc." << std::endl;
  }
}

PinSet SDCParser::getPorts() {
  // [...]
  curToken += 1; // consume "["
  PinSet ports;

  while (!isEndOfTokenStream() && "]" != *curToken) {
    // get_ports [name|{name+}]
    if ("get_ports" == *curToken) {
      curToken += 1; // consume "get_ports"
      if ("{" == *curToken) {
        curToken += 1; // consume "{"
        while (!isEndOfTokenStream() && "}" != *curToken) {
          collectPort(ports, getVarName()); // consume name
        }
        curToken += 1; // consume "}"
      }
      else {
        collectPort(ports, getVarName()); // consume name
      }
    }
    else if ("all_inputs" == *curToken) {
      ports = sdc.m.inPins;
      curToken += 1;
    }
    else if ("all_outputs" == *curToken) {
      ports = sdc.m.outPins;
      curToken += 1; // consume "all_outputs"
    }
    else if ("all_registers" == *curToken) {
      // TODO: extract clock pins of all FFs
      curToken += 1; // consume "all_registers"
    }
    else {
      collectPort(ports, getVarName()); // consume name
    }
  }

  curToken += 1; // consume "]"
  return ports;
}

void SDCParser::tokenizeFile(std::string inName) {
  std::vector<char> delimiters = {'[', ']', '{', '}', '\\', '/', '\"'};
  std::vector<char> separators = {' ', '\t', '\n', '\r'};

  Tokenizer tokenizer(delimiters, separators);
  tokens = tokenizer.tokenize(inName);
  curToken = tokens.begin();
}

void SDCParser::parseClock() {
  curToken += 1; // consume "create_clock"
  auto clk = new SDCClock;
  clk->src = nullptr;
  clk->period = 0.0;
  clk->name = "";

  while(!isEndOfCommand()) {
    // -period value
    if ("-period" == *curToken) {
      curToken += 1; // consume "-period"
      clk->period = getMyFloat(*curToken);
      curToken += 1; // consume value
    }
    // -name name
    else if ("-name" == *curToken) {
      curToken += 1; // consume "-name"
      clk->name = getVarName(); // consume name
    }
    else if ("-add" == *curToken) {
      curToken += 1; // consume "-add"
    }
    else if ("[" == *curToken) {
      auto ports = getPorts();
      if (!ports.empty()) {
        clk->src = *(ports.begin());
      }
    }
    // skip trailing "\\"
    else if ("\\" == *curToken) {
      curToken += 1;
    }
    else {
      std::cout << "Abort: unsupported attribute " << *curToken << " in sdc command create_clock." << std::endl;
      delete clk;
      std::abort();
    }
  }

  // clean up waveform
  auto& wv = clk->waveform;
  if (wv.empty()) {
    wv.push_back({0.0, true});
    wv.push_back({(clk->period / (MyFloat)2.0), false});
  }
  else {
    std::sort(wv.begin(), wv.end(),
        [&] (const SDCClockEdge& e1, const SDCClockEdge& e2) -> bool {
          return (e1.t < e2.t);
        }
    );
  }

  // clean up naming
  if (clk->name.empty()) { 
    if (!(clk->src)) {
      clk->name = clk->src->name;
    }
    else {
      std::cout << "Abort: unnamed virtual clock in sdc." << std::endl;
      delete clk;
      std::abort();
    }
  }

  sdc.clocks[clk->name] = clk;
}

void SDCParser::parseDrivingCell() {
  // set_driving_cell -lib_cell cell -pin pin [port_list] -input_transition_fall value -input_transition_rise value
  curToken += 1; // consume "set_driving_cell"

  PinSet ports;
  Cell* cell = nullptr;
  auto dCell = sdc.addDrivingCell();
  dCell->toCellPin = nullptr;
  dCell->fromCellPin = nullptr;
  dCell->slew[0] = 0.0;
  dCell->slew[1] = 0.0;

  while (!isEndOfCommand()) {
    if ("-lib_cell" == *curToken) {
      curToken += 1; // consume "-lib_cell"
      cell = curLib->findCell(getVarName());
    }
    else if ("-pin" == *curToken) {
      curToken += 1; // consume "-pin"
      dCell->toCellPin = cell->findCellPin(getVarName());
    }
    else if ("-from_pin" == *curToken) {
      curToken += 1; // consume "-from_pin"
      dCell->fromCellPin = cell->findCellPin(getVarName());
    }
    else if ("[" == *curToken) {
      ports = getPorts();
    }
    else if ("-input_transition_fall" == *curToken) {
      curToken += 1; // consume "-input_transition_fall"
      dCell->slew[0] = getMyFloat(*curToken);
      curToken += 1; // consume value
    }
    else if ("-input_transition_rise" == *curToken) {
      curToken += 1; // consume "-input_transition_rise"
      dCell->slew[1] = getMyFloat(*curToken);
      curToken += 1; // consume value
    }
    // skip trailing "\\"
    else if ("\\" == *curToken) {
      curToken += 1;
    }
    else {
      std::cout << "Abort: unsupported attribute " << *curToken << " in sdc command set_driving_cell." << std::endl;
      std::abort();
    }
  }

  if (nullptr == dCell->fromCellPin) {
    dCell->fromCellPin = cell->inPins.begin()->second;
  }

  for (auto& i: ports) {
    sdc.attachPin2DrivingCell(i, dCell);
  }
}

void SDCParser::parseLoad() {
  // set_load -pin_load value [port_list]
  curToken += 1; // consume "set_load"

  MyFloat value = 0.0;
  PinSet ports;
  while (!isEndOfCommand()) {
    if ("-pin_load" == *curToken) {
      curToken += 1; // consume "-pin_load"
      value = getMyFloat(*curToken);
      curToken += 1; // consume value
    }
    else if ("[" == *curToken) {
      ports = getPorts();
    }
    // skip trailing "\\"
    else if ("\\" == *curToken) {
      curToken += 1;
    }
    else {
      std::cout << "Abort: unsupported attribute " << *curToken << " in sdc command create_clock." << std::endl;
      std::abort();
    }
  }

  for (auto& i: ports) {
    sdc.pinLoads[i] = value;
  }
}

void SDCParser::parseMaxDelay() {
  // set_max_delay value -from [port_list] -to [port_list]
  curToken += 1; // consume "set_max_delay"

  MyFloat value = getMyFloat(*curToken);
  curToken += 1; // consume value

  PinSet fromPins, toPins;
  while (!isEndOfCommand()) {
    if ("-from" == *curToken) {
      curToken += 1; // consume "-from"
      fromPins = getPorts();
    }
    else if ("-to" == *curToken) {
      curToken += 1;
      toPins = getPorts();
    }
    // skip trailing "\\"
    else if ("\\" == *curToken) {
      curToken += 1;
    }
    else {
      std::cout << "Abort: unsupported attribute " << *curToken << " in sdc command set_max_delay." << std::endl;
      std::abort();
    }
  }

  if ((fromPins == sdc.m.inPins) && (toPins == sdc.m.outPins)) {
    sdc.maxDelayPI2PO = value;
  }
  // TODO: handle register cases
}

Token SDCParser::getVarName() {
  Token t = *curToken;
  curToken += 1;

  if ("\\" == t) {
    t += *curToken;
    curToken += 1;
  }

  // variable in the form of a\[2\]
  if ("\\" == *curToken) {
    curToken += 1; // skip "\\"

    t += *curToken; // get "["
    curToken += 1; // consume "["

    for ( ; "]" != *curToken; curToken += 1) {
      if ("\\" != *curToken) {
        t += *curToken;
      }
    }

    t += *curToken; // get "]"
    curToken += 1; // consume "]"
  }

  return t;
}

void SDCParser::parse(std::string inName) {
  tokenizeFile(inName);
  defaultLib = *(sdc.libs.begin());
  std::cout << defaultLib->name << " is used" << std::endl;

  while (!isEndOfTokenStream()) {
    curLib = defaultLib;

    if ("create_clock" == *curToken) {
      parseClock();
    }
    else if ("set_max_delay" == *curToken) {
      parseMaxDelay();
    }
    else if ("set_driving_cell" == *curToken) {
      parseDrivingCell();
    }
    else if ("set_load" == *curToken) {
      parseLoad();
    }
    else {
      std::cout << "Abort SDC: unsupported command " << *curToken << "." << std::endl;
      return;
    }
  }
}

void SDC::clear() {
  for (auto i: drivingCells) {
    delete i;
  }
  drivingCells.clear();

  for (auto& i: clocks) {
    delete i.second;
  }
  clocks.clear();
}

void SDC::parse(std::string inName, bool toClear) {
  if (inName.empty()) {
    std::cout << "No sdc specified. Sdc remains unchanged." << std::endl;
    return;
  }

  if (toClear) {
    clear();
  }
  SDCParser parser(*this);
  parser.parse(inName);
}

void SDC::print(std::ostream& os) {
  os << "SDC max delays:" << std::endl;
  os << "  PI 2 PO: " << maxDelayPI2PO << std::endl;
  os << "  PI 2 RI: " << maxDelayPI2RI << std::endl;
  os << "  RO 2 RI: " << maxDelayRO2RI << std::endl;
  os << "  RO 2 PO: " << maxDelayRO2PO << std::endl;

  if (!clocks.empty()) {
    os << "SDC clocks:" << std::endl;
    for (auto& i: clocks) {
      i.second->print(os);
    }
  }

  if (!mapPin2DrivingCells.empty()) {
    os << "SDC driving cells:" << std::endl;
    for (auto& i: mapPin2DrivingCells) {
      os << "  Driving cell of " << i.first->name << ":" << std::endl;
      i.second->print(os);
    }
  }

  if (!pinLoads.empty()) {
    os << "SDC pin loads:" << std::endl;
    for (auto& i: pinLoads) {
      os << "  Load of " << i.first->name << " = " << i.second << std::endl;
    }
  }
}

void SDCClock::print(std::ostream& os) {
  os << "  Clock " << name << ":" << std::endl;
  os << "    period = " << period << std::endl;
  os << "    src pin = " << ((src) ? src->name : "(virtual)") << std::endl;
  os << "    wave from = { ";
  for (auto& i: waveform) {
    i.print(os);
    os << " ";
  }
  os << "}" << std::endl;
}

void SDCClockEdge::print(std::ostream& os) {
  os << "(" << t << ", " << ((isRise) ? "r" : "f") << ")";
}

void SDCDrivingCell::print(std::ostream& os) {
  std::string cellName = toCellPin->cell->name;
  os << "    pin = " << cellName << "." << toCellPin->name << std::endl;
  os << "    from_pin = " << cellName << "." << fromCellPin->name << std::endl;
  os << "    input_transition_fall = " << slew[0] << std::endl;
  os << "    input_transition_rise = " << slew[1] << std::endl;
}
