#include "Sdc.h"

#include <cstdlib>
#include <algorithm>

using PinSet = std::unordered_set<VerilogPin*>;
static const MyFloat infinity = std::numeric_limits<MyFloat>::infinity();

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
  std::vector<char> separators = {' ', '\t', '\n', '\r', ';'};
  std::vector<std::string> comments = {"#", "\n"};

  Tokenizer tokenizer(delimiters, separators, comments);
  tokens = tokenizer.tokenize(inName);
  curToken = tokens.begin();
}

void SDCParser::parseCreateClock() {
  curToken += 1; // consume "create_clock"
  auto clk = new Clock;
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
        [&] (const ClockEdge& e1, const ClockEdge& e2) -> bool {
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

void SDCParser::parseSetLoad() {
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
    sdc.getEnvAtPort(i)->outputLoad = value;
  }
}

void SDCParser::parseSetInputDelay() {
  // set_input_delay value -max/min -fall/rise [port_list] -clock clk_name
  curToken += 1; // consume "set_input_delay"

  MyFloat value = getMyFloat(*curToken);
  curToken += 1;

  PinSet ports;
  TimingMode mode = MAX_DELAY_MODE;
  bool isRise = false;
  Clock* clk = nullptr;

  while (!isEndOfCommand()) {
    if ("-max" == *curToken) {
      mode = MAX_DELAY_MODE;
      curToken += 1;
    }
    else if ("-min" == *curToken) {
      mode = MIN_DELAY_MODE;
      curToken += 1;
    }
    else if ("-fall" == *curToken) {
      isRise = false;
      curToken += 1;
    }
    else if ("-rise" == *curToken) {
      isRise = true;
      curToken += 1;
    }
    else if ("-clock" == *curToken) {
      curToken += 1; // consume "clock"
      clk = sdc.getClock(getVarName()); // consume clk_name
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
    auto env = sdc.getEnvAtPort(i);
    env->inputDelay[mode][isRise] = value;
    env->clk = clk;
    env->pin = i;
  }
}

void SDCParser::parseSetInputTransition() {
  // set_input_transition value -max/min -fall/rise [port_list] -clock clk_name
  curToken += 1; // consume "set_input_transition"

  MyFloat value = getMyFloat(*curToken);
  curToken += 1;

  PinSet ports;
  TimingMode mode = MAX_DELAY_MODE;
  bool isRise = false;
  Clock* clk = nullptr;

  while (!isEndOfCommand()) {
    if ("-max" == *curToken) {
      mode = MAX_DELAY_MODE;
      curToken += 1;
    }
    else if ("-min" == *curToken) {
      mode = MIN_DELAY_MODE;
      curToken += 1;
    }
    else if ("-fall" == *curToken) {
      isRise = false;
      curToken += 1;
    }
    else if ("-rise" == *curToken) {
      isRise = true;
      curToken += 1;
    }
    else if ("-clock" == *curToken) {
      curToken += 1; // consume "clock"
      clk = sdc.getClock(getVarName()); // consume clk_name
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
    auto env = sdc.getEnvAtPort(i);
    env->inputSlew[mode][isRise] = value;
    env->clk = clk;
    env->pin = i;
  }
}

void SDCParser::parseSetOutputDelay() {
  // set_output_delay value -max/min -fall/rise [port_list] -clock clk_name
  curToken += 1; // consume "set_output_delay"

  MyFloat value = getMyFloat(*curToken);
  curToken += 1;

  PinSet ports;
  TimingMode mode = MAX_DELAY_MODE;
  bool isRise = false;
  Clock* clk = nullptr;

  while (!isEndOfCommand()) {
    if ("-max" == *curToken) {
      mode = MAX_DELAY_MODE;
      curToken += 1;
    }
    else if ("-min" == *curToken) {
      mode = MIN_DELAY_MODE;
      curToken += 1;
    }
    else if ("-fall" == *curToken) {
      isRise = false;
      curToken += 1;
    }
    else if ("-rise" == *curToken) {
      isRise = true;
      curToken += 1;
    }
    else if ("-clock" == *curToken) {
      curToken += 1; // consume "clock"
      clk = sdc.getClock(getVarName()); // consume clk_name
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
    auto env = sdc.getEnvAtPort(i);
    env->outputDelay[mode][isRise] = value;
    env->clk = clk;
    env->pin = i;
  }
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

  while (!isEndOfTokenStream()) {
    if ("create_clock" == *curToken) {
      parseCreateClock();
    }
    else if ("set_load" == *curToken) {
      parseSetLoad();
    }
    else if ("set_input_delay" == *curToken) {
      parseSetInputDelay();
    }
    else if ("set_input_transition" == *curToken) {
      parseSetInputTransition();
    }
    else if ("set_output_delay" == *curToken) {
      parseSetOutputDelay();
    }
    else {
      std::cout << "Abort SDC: unsupported command " << *curToken << "." << std::endl;
      return;
    }
  }
}

static std::unordered_map<TimingMode, std::string> mapTMode2Name = {
  {MAX_DELAY_MODE, "-max"},
  {MIN_DELAY_MODE, "-min"}
};

static std::unordered_map<bool, std::string> mapRF2Name = {
  {false, "-fall"},
  {true, "-rise"}
};

void SDCEnvAtPort::print(std::ostream& os) {
  auto printItem =
      [&] (std::string prompt, MyFloat value, TimingMode i, bool j) {
        if (value != infinity) {
          os << prompt << " " << value;
          os << " " << mapTMode2Name[i];
          os << " " << mapRF2Name[j];
          os << " [get_ports " << pin->name << "]";
          os << " -clock " << clk->name << std::endl;
        }
      };

  for (int i = 0; i < 2; i++) {
    TimingMode mode = (TimingMode)i;
    for (int j = 0; j < 2; j++) {
      bool rf = (bool)j;
      printItem("set_input_delay", inputDelay[i][j], mode, rf);
      printItem("set_input_transition", inputSlew[i][j], mode, rf);
      printItem("set_output_delay", outputDelay[i][j], mode, rf);
    }
  }

  if (outputLoad != infinity) {
    os << "set_load -pin_load " << outputLoad;
    os << " [get_ports " << pin->name << "]" << std::endl;
  }
}

void SDC::clear() {
  for (auto& i: clocks) {
    delete i.second;
  }
  clocks.clear();

  for (auto& i: envAtPorts) {
    delete i.second;
  }
  envAtPorts.clear();
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
  for (auto& c: clocks) {
    c.second->print(os);
  }

  for (auto& i: envAtPorts) {
    i.second->print(os);
  }
}
