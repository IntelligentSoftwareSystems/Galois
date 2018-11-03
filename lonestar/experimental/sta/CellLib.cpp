#include <cassert>
#include <algorithm>

#include "CellLib.h"

using Bound = std::pair<size_t, size_t>;

template<typename T>
static Bound findBound(T v, std::vector<T>& array) {
  // array should be non-empty
  auto arraySize = array.size();
  assert(arraySize);

  if (1 == arraySize) {
    return {0, 0};
  }
  else if (2 == arraySize) {
    return {0, 1};
  }
  else {
    auto upper = std::upper_bound(array.begin(), array.end(), v);
    // all elements in array are smaller than v
    if (upper == array.end()) {
      return {array.size() - 2, array.size() - 1};
    }
    // all elements in array are smaller than v
    else if (upper == array.begin()) {
      return {0, 1};
    }
    // v sits in between a pair of elements in array
    else {
      auto index_u = std::distance(array.begin(), upper);
      return {index_u - 1, index_u};
    }
  }
}

template<typename T>
static T interpolate(T x0, T y0, T x1, T y1, T x) {
  if (x1 == x0) {
    assert(y1 == y0);
    return y0;
  }
  else {
    auto xDiff = x1 - x0;
    auto yDiff = y1 - y0;
    return y0 + (x - x0) * yDiff / xDiff;
  }
}

MyFloat Lut::lookupInternal(VecOfMyFloat& param, std::vector<Bound>& bound, std::vector<size_t>& stride, size_t start, size_t lv) {
  auto lb = bound[lv].first;
  auto ub = bound[lv].second;
  auto start_l = start + stride[lv] * lb;
  auto start_u = start + stride[lv] * ub;
  bool isBottomMost = (lv == bound.size() - 1);
  auto y0 = (isBottomMost) ? value[start_l] : lookupInternal(param, bound, stride, start_l, lv+1);
  auto y1 = (isBottomMost) ? value[start_u] : lookupInternal(param, bound, stride, start_u, lv+1);
  return interpolate(index[lv][lb], y0, index[lv][ub], y1, param[lv]);
}

MyFloat Lut::lookup(Parameter& param) {
  auto paramSize = param.size();
  assert(paramSize == index.size());

  // scalar case, return the value immediately
  if (0 == paramSize) {
    return value[0];
  }

  // sort parameter by the var order in lutTemplate
  // assume no repeated variables
  VecOfMyFloat sortedParam;
  for (size_t i = 0; i < paramSize; ++i) {
    sortedParam.push_back(param.at(lutTemplate->var[i]));
  }

  // find bounds for each dimension
  std::vector<Bound> bound;
  for (size_t i = 0; i < paramSize; ++i) {
    bound.push_back(findBound(sortedParam[i], index[i]));
  }

  return lookupInternal(sortedParam, bound, lutTemplate->stride, 0, 0);
}

void Lut::print(std::string attr, std::ostream& os) {
  os << "        " << attr << " (" << lutTemplate->name << ") {" << std::endl;

  for (size_t i = 0; i < index.size(); i++) {
    os << "          index_" << (i+1) << " (\"" << index[i][0];
    for (size_t j = 1; j < index[i].size(); j++) {
      os << ", " << index[i][j];
    }
    os << "\");" << std::endl;
  }

  auto numForLine = *(lutTemplate->shape.rbegin());
  auto numForTable = value.size();

  for (size_t i = 0; i < value.size(); i++) {
    if (0 == i) {
      os << "          values (\"";
    }
    else if (0 == i % numForLine) {
      os << "                  \"";
    }

    os << value[i];

    if ((numForLine - 1) != i % numForLine) {
      os << ", ";
    }
    else {
      os << "\"";
      if (i != numForTable - 1) {
        os << ", \\";
      }
      else {
        os << ");";
      }
      os << std::endl;
    }
  }

  os << "        }" << std::endl;
}

void Lut::wrapUpConstruction() {
  // use index from the template when not specified
  if (0 == index.size()) {
    index = lutTemplate->index;
  }
}

TimingTable::~TimingTable() {
  for (int i = 0; i < 2; i++) {
    auto t = delay[i];
    if (t) { delete t; }
    t = slew[i];
    if (t) { delete t; }
    t = constraint[i];
    if (t) { delete t; }
  }
}

static std::unordered_map<TimingSense, std::string> mapTSense2Name = {
  {POSITIVE_UNATE, "positive_unate"},
  {NEGATIVE_UNATE, "negative_unate"},
  {NON_UNATE,      "non_unate"},
};

static std::unordered_map<TimingType, std::string> mapTType2Name = {
  {COMBINATIONAL,             "combinational"},
  {COMBINATIONAL_RISE,        "combinational_rise"},
  {COMBINATIONAL_FALL,        "combinational_fall"},
  {THREE_STATE_DISABLE,       "three_state_disable"},
  {THREE_STATE_DISABLE_RISE,  "three_state_disable_rise"},
  {THREE_STATE_DISABLE_FALL,  "three_state_disable_fall"},
  {THREE_STATE_ENABLE,        "three_state_enable"},
  {THREE_STATE_ENABLE_RISE,   "three_state_enable_rise"},
  {THREE_STATE_ENABLE_FALL,   "three_state_enable_fall"},
  {RISING_EDGE,               "rising_edge"},
  {FALLING_EDGE,              "falling_edge"},
  {PRESET,                    "preset"},
  {CLEAR,                     "clear"},
  {HOLD_RISING,               "hold_rising"},
  {HOLD_FALLING,              "hold_falling"},
  {SETUP_RISING,              "setup_rising"},
  {SETUP_FALLING,             "setup_falling"},
  {RECOVERY_RISING,           "recovery_rising"},
  {RECOVERY_FALLING,          "recovery_falling"},
  {SKEW_RISING,               "skew_rising"},
  {SKEW_FALLING,              "skew_falling"},
  {REMOVAL_RISING,            "removal_rising"},
  {REMOVAL_FALLING,           "removal_falling"},
  {MIN_PULSE_WIDTH,           "min_pulse_width"},
  {MINIMUM_PERIOD,            "minimum_period"},
  {MAX_CLOCK_TREE_PATH,       "max_clock_tree_path"},
  {MIN_CLOCK_TREE_PATH,       "min_clock_tree_path"},
  {NON_SEQ_SETUP_RISING,      "non_seq_setup_rising"},
  {NON_SEQ_SETUP_FALLING,     "non_seq_setup_falling"},
  {NON_SEQ_HOLD_RISING,       "non_seq_hold_rising"},
  {NON_SEQ_HOLD_FALLING,      "non_seq_hold_falling"},
  {NOCHANGE_HIGH_HIGH,        "nochange_high_high"},
  {NOCHANGE_HIGH_LOW,         "nochange_high_low"},
  {NOCHANGE_LOW_HIGH,         "nochange_low_high"},
  {NOCHANGE_LOW_LOW,          "nochange_low_low"},
};

void TimingTable::print(std::ostream& os) {
  os << "      timing () {" << std::endl;
  os << "        related_pin: \"" << relatedPin->name << "\";" << std::endl;

  // do not show combinational (default value)
  if (COMBINATIONAL != tType) {
    os << "        timing_type: " << mapTType2Name.at(tType) << ";" << std::endl;
  }
  if (!when.empty()) {
    os << "        when: \"" << when << "\";" << std::endl;
  }
  // only delay arcs will show timing sense
  if (unate != NOT_APPLICABLE) {
    os << "        timing_sense: " << mapTSense2Name.at(unate) << ";" << std::endl;
  }

  for (int i = 0; i < 2; i++) {
    std::string direction = (i) ? "rise" : "fall";
    if (delay[i]) {
      delay[i]->print("cell_" + direction, os);
    }
    if (slew[i]) {
      slew[i]->print(direction + "_transition", os);
    }
    if (constraint[i]) {
      constraint[i]->print(direction + "_constraint", os);
    }
  }

  os << "      }" << std::endl;
}

void TimingTable::wrapUpConstruction() {
  // lookup for the pin now
  // because not all dependent pins show up before the table when parsing
  assert(!nameOfRelatedPin.empty());
  relatedPin = endPin->cell->findCellPin(nameOfRelatedPin);
  assert(relatedPin);

  // relate tables to lookup structure
}

PowerTable::~PowerTable() {
  for (int i = 0; i < 2; i++) {
    auto t = internalPower[i];
    if (t) { delete t; }
  }
}

void PowerTable::wrapUpConstruction() {
  // find the pin now
  // because not all dependent pins show up before the table when parsing
  if (!nameOfRelatedPin.empty()) {
    relatedPin = endPin->cell->findCellPin(nameOfRelatedPin);
    assert(relatedPin);
  }
}

void PowerTable::print(std::ostream& os) {
  os << "      internal_power () {" << std::endl;
  if (relatedPin) {
    os << "        related_pin: " << relatedPin->name << ";" << std::endl;
  }
  if (!when.empty()) {
    os << "        when: \"" << when << "\";" << std::endl;
  }

  for (int i = 0; i < 2; i++) {
    std::string direction = (i) ? "rise" : "fall";
    auto pt = internalPower[i];
    if (pt) {
      pt->print(direction + "_power", os);
    }
  }

  os << "      }" << std::endl;
}

bool CellPin::isEdgeDefined(CellPin* inPin, bool isInRise, bool isMeRise, TableType index) {
  return true;
}

MyFloat CellPin::extract(Parameter& param, TableType index, CellPin* inPin, bool isInRise, bool isMeRise, std::string when) {
  if (POWER == index) {
    return 0.0;
  }
  else {
    return 0.0;
  }
}

std::pair<MyFloat, std::string>
CellPin::extractMax(Parameter& param, TableType index, CellPin* inPin, bool isInRise, bool isMeRise) {
  MyFloat ret = -std::numeric_limits<MyFloat>::infinity();
  std::string when;
  return {ret, when};
}

std::pair<MyFloat, std::string>
CellPin::extractMin(Parameter& param, TableType index, CellPin* inPin, bool isInRise, bool isMeRise) {
  MyFloat ret = std::numeric_limits<MyFloat>::infinity();
  std::string when;
  return {ret, when};
}

static std::unordered_map<PinDirection, std::string> mapPinDir2Name = {
  {INPUT,    "input"},
  {OUTPUT,   "output"},
  {INOUT,    "inout"},
  {INTERNAL, "internal"}
};

void CellPin::print(std::ostream& os) {
  os << "    pin (" << name << ") {" << std::endl;
  os << "      direction: " << mapPinDir2Name.at(dir) << ";" << std::endl;

  if (INPUT == dir || INOUT == dir) {
    if (isClock) {
      os << "      clock: true;" << std::endl;
    }
    os << "      capactance: " << ((c[1] > c[0]) ? c[1] : c[0]) << ";" << std::endl;
    os << "      rise_capacitance: " << c[1] << ";" << std::endl;
    os << "      fall_capacitance: " << c[0] << ";" << std::endl;
  }

  if (OUTPUT == dir || INOUT == dir) {
    os << "      max_capacitance: " << maxC << ";" << std::endl;
    if (func.size()) {
      os << "      function: \"" << func << "\";" << std::endl;
    }
    if (func_up.size()) {
      os << "      function_up: \"" << func_up << "\";" << std::endl;
    }
    if (func_down.size()) {
      os << "      function_down: \"" << func_down << "\";" << std::endl;
    }
  }

  for (auto& i: timings) {
    i->print(os);
  }

  for (auto& i: powers) {
    i->print(os);
  }

  os << "    }" << std::endl;
}

void CellPin::wrapUpConstruction() {
  for (auto& i: timings) {
    i->wrapUpConstruction();
  }
  for (auto& i: powers) {
    i->wrapUpConstruction();
  }
}

CellPin::~CellPin() {
  for (auto& i: timings) {
    delete i;
  }
  for (auto& i: powers) {
    delete i;
  }
}

void Cell::print(std::ostream& os) {
  os << "  cell (" << name << ") {" << std::endl;

  os << "    drive_strength: " << driveStrength << ";" << std::endl;
  os << "    area: " << area << ";" << std::endl;
  os << "    cell_leakage_power: " << cellLeakagePower << ";" << std::endl;

  for (auto& i: leakagePower) {
    os << "    leakage_power () {" << std::endl;
    os << "      when: \"" << i.first << "\";" << std::endl;
    os << "      value: " << i.second << ";" << std::endl;
    os << "    }" << std::endl;
  }

  for (auto& i: inPins) {
    i->print(os);
  }

  for (auto& i: internalPins) {
    i->print(os);
  }

  for (auto& i: outPins) {
    i->print(os);
  }

  os << "  }" << std::endl;
}

void Cell::wrapUpConstruction() {
  for (auto& i: pins) {
    i.second->wrapUpConstruction();
  }
}

Cell::~Cell() {
  for (auto& i: pins) {
    delete i.second;
  }
}

static std::unordered_map<VariableType, std::string> mapVarType2Name = {
  {INPUT_TRANSITION_TIME,        "input_transition_time"},
  {CONSTRAINED_PIN_TRANSITION,   "constrained_pin_transition"},
  {RELATED_PIN_TRANSITION,       "related_pin_transition"},
  {TOTAL_OUTPUT_NET_CAPACITANCE, "total_output_net_capacitance"},
  {INPUT_NET_TRANSITION,         "input_net_transition"},
  {TIME,                         "time"}
};

void LutTemplate::wrapUpConstruction() {
  // find stride for each dimension
  stride.insert(stride.begin(), shape.size(), 1);
  for (size_t i = shape.size() - 1; i >= 1; --i) {
    stride[i-1] = stride[i] * shape[i];
  }
}

void LutTemplate::print(std::ostream& os) {
  if ("scalar" == name) { return; }

  std::string attr = (isForPower) ? "power_lut_template" : "lu_table_template";
  os << "  " << attr << " (" << name << ") {" << std::endl;
  size_t i = 0;
  for (auto& v: var) {
    os << "    variable_" << ++i << ": " << mapVarType2Name.at(v) << ";" << std::endl;
  }

  i = 0;
  for (auto& idx: index) {
    os << "    index_" << ++i << " (\"";
    for (size_t j = 0; j < idx.size(); j++) {
      os << idx[j];
      if (j < idx.size() - 1) {
        os << ", ";
      }
    }
    os << "\");" << std::endl;
  }
#if 0
  os << "    shape:";
  i = 0;
  for (auto& v: shape) {
    os << " " << v;
  }
  os << std::endl;

  os << "    stride:";
  i = 0;
  for (auto& v: stride) {
    os << " " << v;
  }
  os << std::endl;
#endif
  os << "  }" << std::endl;
}

MyFloat PreLayoutWireLoad::wireLength(size_t deg) {
  auto lb = fanoutLength.lower_bound(deg);
  // extrapolation from upper limit
  if (lb == fanoutLength.end()) {
    auto last = fanoutLength.rbegin();
    return last->second + (MyFloat)(deg - last->first) * slope;
  }
  // exact match
  else if (lb->first == deg) {
    return lb->second;
  }
  // interpolation
  else {
    auto ub = fanoutLength.begin();
    if (lb == ub) {
      ub++;
    }
    else {
      ub = lb;
      lb--;
    }
    return interpolate((MyFloat)lb->first, lb->second, (MyFloat)ub->first, ub->second, (MyFloat)deg);
  }
}

MyFloat PreLayoutWireLoad::wireC(VerilogWire* wire) {
  return c * wireLength(wire->outDeg());
}

MyFloat PreLayoutWireLoad::wireDelay(MyFloat loadC, VerilogWire* wire, VerilogPin* vPin) {
  // best-case tree
  if (BEST_CASE_TREE == lib->wireTreeType) {
    return 0.0;
  }
  else {
    auto deg = wire->outDeg();
    auto wL = wireLength(deg);
    auto wC = c * wL;
    auto wR = r * wL;

    // balanced tree
    if (BALANCED_TREE == lib->wireTreeType) {
      wC /= (MyFloat)deg;
      wR /= (MyFloat)deg;
    }

    // Elmore delay for worst-case & balanced tree
    // Ref: J. Bhasker, R. Chadha. STA for nanometer designs: a practical approach
    //      Springer, 2009.
//      auto delay = wR * (wC / 2 + loadC);

    // delay formula used by Cadence Genus' report_net_delay_calculation
    auto delay = wR * (wC + loadC);

    return delay;
  }
}

void PreLayoutWireLoad::print(std::ostream& os) {
  os << "  wire_load (\"" << name << "\") {" << std::endl;
  os << "    capacitance: " << c << ";" << std::endl;
  os << "    resistance: " << r << ";" << std::endl;
  os << "    slope: " << slope << ";" << std::endl;
  for (auto& i: fanoutLength) {
    os << "    fanout_length (" << i.first << ", " << i.second << ");" << std::endl;
  }
  os << "  }" << std::endl;
}

void CellLibParser::skipAttribute() {
  while (!isEndOfTokenStream()) {
    if (";" != *curToken) {
      ++curToken;
    }
    else {
      ++curToken; // consume ";"
      break;
    }
  }
}

void CellLibParser::skipGroup() {
  while (!isEndOfTokenStream() && "{" != *curToken) {
    ++curToken;
  }
  ++curToken; // consume "{"

  // skip all statements in this group statement
  while (!isEndOfGroup()) {
    skip(false);
  }
  ++curToken; // consume "}"
}

void CellLibParser::skip(bool isTopCall) {
  if (isEndOfTokenStream()) {
    return;
  }

  // look ahead for statement types
  bool skipped = false;
  for (auto nextToken = curToken + 1; nextToken != tokens.end(); ++nextToken) {
    // simple attributes
    // attr_key_word: value;
    if (":" == *nextToken) {
#if 0
      if (isTopCall) {
        std::cout << "Skip attribute " << *curToken << std::endl;
      }
#endif
      skipAttribute();
      skipped = true;
      break;
    }
    // group statements
    // group_key_word (...) {...}
    else if ("{" == *nextToken) {
#if 0
      if (isTopCall) {
        std::cout << "Skip group statement " << *curToken << std::endl;
      }
#endif
      skipGroup();
      skipped = true;
      break;
    }
    // complex attributes
    // attr_key_word (...);
    else if (";" == *nextToken) {
#if 0
      if (isTopCall) {
        std::cout << "Skip attribute " << *curToken << std::endl;
      }
#endif
      skipAttribute();
      skipped = true;
      break;
    }
  }

  // skip to end if no legal statements to skip
  if (!skipped) {
    curToken = tokens.end();
  }
}

void CellLibParser::parseWireLoad() {
  // wire_load ("name") {...}
  curToken += 3; // consume "wire_load", "\"", and "("
  auto wireLoad = lib->addWireLoad(*curToken);
  curToken += 4; // consume name, "\"", ")", and "{"

  while (!isEndOfGroup()) {
    // capacitance: value;
    if ("capacitance" == *curToken) {
      curToken += 2; // consume "capacitance" and ":"
      wireLoad->c = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    // resisance: value;
    else if ("resistance" == *curToken) {
      curToken += 2; // consume "resistance" and ":"
      wireLoad->r = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    // slope: value;
    else if ("slope" == *curToken) {
      curToken += 2; // consume "slope" and ":"
      wireLoad->slope = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    // fanout_length(fanout, length);
    else if ("fanout_length" == *curToken) {
      curToken += 2; // consume "fanout_length" and "("
      size_t fanout = std::stoul(*curToken);
      curToken += 2; // consume fanout and ","
      MyFloat length = getMyFloat(*curToken);
      wireLoad->addFanoutLength(fanout, length);
      curToken += 3; // consume length, ")" and ";"
    }
    else {
      skip();
    }
  }

  curToken += 1; // consume "}"
}

static std::unordered_map<std::string, VariableType> mapName2VarType = {
  {"input_transition_time",        INPUT_TRANSITION_TIME},
  {"constrained_pin_transition",   CONSTRAINED_PIN_TRANSITION},
  {"related_pin_transition",       RELATED_PIN_TRANSITION},
  {"total_output_net_capacitance", TOTAL_OUTPUT_NET_CAPACITANCE},
  {"input_net_transition",         INPUT_NET_TRANSITION},
  {"time",                         TIME}
};

void CellLibParser::parseLutTemplate(bool isForPower) {
  // lu_table_template/power_lut_template (name) {...}
  curToken += 2; // consume "lu_table_tamplate/power_lut_template" and "("
  auto lutTemplate = lib->addLutTemplate(*curToken, isForPower);
  curToken += 3; // consume name, ")", and "{"

  int unmatched = 0;
  while (!isEndOfGroup()) {
    // variable_*: name;
    if ("variable_1" == *curToken || "variable_2" == *curToken || "variable_3" == *curToken) {
      curToken += 2; // consume "variable_*" and ":"
      lutTemplate->var.push_back(mapName2VarType.at(*curToken));
      curToken += 2; // consume name and ";"
      unmatched++;
    }
    // index_* ("num1[,num*]");
    else if ("index_1" == *curToken || "index_2" == *curToken || "index_3" == *curToken) {
      curToken += 3; // consume "index_*", "(", "\""
      VecOfMyFloat oneIndex;
      oneIndex.push_back(getMyFloat(*curToken));
      curToken += 1; // consume value
      while ("," == *curToken) {
        curToken += 1; // consume ","
        oneIndex.push_back(getMyFloat(*curToken));
        curToken += 1; // consume value
      }
      lutTemplate->shape.push_back(oneIndex.size());
      lutTemplate->index.push_back(oneIndex);
      curToken += 3; // consume "\"", ")" and ";"
      unmatched--;
    }
    else {
      skip();
    }
  }

  assert(!unmatched);
  curToken += 1; // consume "}"
  lutTemplate->wrapUpConstruction();
}

void CellLibParser::parseLut(Lut* lut, bool isForPower) {
  // key_word (name) {...}
  curToken += 2; // consume key_word and "("
  lut->lutTemplate = lib->findLutTemplate(*curToken, isForPower);
  assert(lut->lutTemplate);
  curToken += 3; // consume name, ")" and "{"

  while (!isEndOfGroup()) {
    // index_* ("num1[,num*]");
    if ("index_1" == *curToken || "index_2" == *curToken || "index_3" == *curToken) {
      curToken += 3; // consume "index_*", "(" and "\""
      VecOfMyFloat v;
      while (!isEndOfTokenStream() && ")" != *curToken) {
        v.push_back(getMyFloat(*curToken));
        curToken += 2; // consume num*, and "," in between or "\"" at the end
      }
      lut->index.push_back(v);
      curToken += 2; // consume ")" and ";"
    }
    // values ("num1[,num*]"[, \ "num1[,num*]"]);
    else if ("values" == *curToken) {
      curToken += 3; // consume "value", "(" and "\""
      while (!isEndOfTokenStream() && ")" != *curToken) {
        lut->value.push_back(getMyFloat(*curToken));
        curToken += 2;
        if ("," == *curToken) {
          curToken += 3; // consume ",", "\\" and "\""
        }
      }
      curToken += 2; // consume ")" and ";"
    }
    else {
      skip();
    }
  }

  curToken += 1; // consume "}"
  lut->wrapUpConstruction();
}

static std::unordered_map<std::string, TimingSense> mapName2TSense = {
  {"positive_unate", POSITIVE_UNATE},
  {"negative_unate", NEGATIVE_UNATE},
  {"non_unate",      NON_UNATE}
};

static std::unordered_map<std::string, TimingType> mapName2TType = {
  {"combinational",            COMBINATIONAL},
  {"combinational_rise",       COMBINATIONAL_RISE},
  {"combinational_fall",       COMBINATIONAL_FALL},
  {"three_state_disable",      THREE_STATE_DISABLE},
  {"three_state_disable_rise", THREE_STATE_DISABLE_RISE},
  {"three_state_disable_fall", THREE_STATE_DISABLE_FALL},
  {"three_state_enable",       THREE_STATE_ENABLE},
  {"three_state_enable_rise",  THREE_STATE_ENABLE_RISE},
  {"three_state_enable_fall",  THREE_STATE_ENABLE_FALL},
  {"rising_edge",              RISING_EDGE},
  {"falling_edge",             FALLING_EDGE},
  {"preset",                   PRESET},
  {"clear",                    CLEAR},
  {"hold_rising",              HOLD_RISING},
  {"hold_falling",             HOLD_FALLING},
  {"setup_rising",             SETUP_RISING},
  {"setup_falling",            SETUP_FALLING},
  {"recovery_rising",          RECOVERY_RISING},
  {"recovery_falling",         RECOVERY_FALLING},
  {"skew_rising",              SKEW_RISING},
  {"skew_falling",             SKEW_FALLING},
  {"removal_rising",           REMOVAL_RISING},
  {"removal_falling",          REMOVAL_FALLING},
  {"min_pulse_width",          MIN_PULSE_WIDTH},
  {"minimum_period",           MINIMUM_PERIOD},
  {"max_clock_tree_path",      MAX_CLOCK_TREE_PATH},
  {"min_clock_tree_path",      MIN_CLOCK_TREE_PATH},
  {"non_seq_setup_rising",     NON_SEQ_SETUP_RISING},
  {"non_seq_setup_falling",    NON_SEQ_SETUP_FALLING},
  {"non_seq_hold_rising",      NON_SEQ_HOLD_RISING},
  {"non_seq_hold_falling",     NON_SEQ_HOLD_FALLING},
  {"nochange_high_high",       NOCHANGE_HIGH_HIGH},
  {"nochange_high_low",        NOCHANGE_HIGH_LOW},
  {"nochange_low_high",        NOCHANGE_LOW_HIGH},
  {"nochange_low_low",         NOCHANGE_LOW_LOW},
};

void CellLibParser::parseTiming(TimingTable* tTable) {
  // timing () {...}
  curToken += 4; // consume "timing", "(", ")" and "{"

  while (!isEndOfGroup()) {
    if ("cell_fall" == *curToken) {
      Lut* lut = new Lut;
      tTable->delay[0] = lut;
      parseLut(lut);
    }
    else if ("cell_rise" == *curToken) {
      Lut* lut = new Lut;
      tTable->delay[1] = lut;
      parseLut(lut);
    }
    else if ("fall_transition" == *curToken) {
      Lut* lut = new Lut;
      tTable->slew[0] = lut;
      parseLut(lut);
    }
    else if ("rise_transition" == *curToken) {
      Lut* lut = new Lut;
      tTable->slew[1] = lut;
      parseLut(lut);
    }
    else if ("fall_constraint" == *curToken) {
      Lut* lut = new Lut;
      tTable->constraint[0] = lut;
      parseLut(lut);
    }
    else if ("rise_constraint" == *curToken) {
      Lut* lut = new Lut;
      tTable->constraint[1] = lut;
      parseLut(lut);
    }
    // when: "...";
    else if ("when" == *curToken) {
      curToken += 2; // consume "when" and ":"
      tTable->when = getBooleanExpression();
      curToken += 1; // consume ";"
    }
    // related_pin: "name";
    else if ("related_pin" == *curToken) {
      curToken += 3; // consume "related_pin", ":" and "\""
      tTable->nameOfRelatedPin = *curToken;
      curToken += 3; // consume name, "\"" and ";"
    }
    // timing_sense: value;
    else if ("timing_sense" == *curToken) {
      curToken += 2; // consume "timing_sense" and ":"
      tTable->unate = mapName2TSense.at(*curToken);
      curToken += 2; // consume value and ";"
    }
    // timing_type: value;
    else if ("timing_type" == *curToken) {
      curToken += 2; // consume "timing_sense" and ":"
      tTable->tType = mapName2TType.at(*curToken);
      curToken += 2; // consume value and ";"
    }
    else {
      skip();
    }
  } // end while

  curToken += 1; // consume "}"
}

void CellLibParser::parseInternalPower(PowerTable* pTable) {
  // internal_power () {...}
  curToken += 4; // consume "internal_power", "(", ")" and "{"

  while (!isEndOfGroup()) {
    if ("fall_power" == *curToken) {
      Lut* lut = new Lut;
      pTable->internalPower[0] = lut;
      parseLut(lut, true);
    }
    else if ("rise_power" == *curToken) {
      Lut* lut = new Lut;
      pTable->internalPower[1] = lut;
      parseLut(lut, true);
    }
    // when: "...";
    else if ("when" == *curToken) {
      curToken += 2; // consume "when" and ":"
      pTable->when = getBooleanExpression();
      curToken += 1; // consume ";"
    }
    // related_pin: "name";
    else if ("related_pin" == *curToken) {
      curToken += 3; // consume "related_pin", ":" and "\""
      pTable->nameOfRelatedPin = *curToken;
      curToken += 3; // consume name, "\"" and ";"
    }
    else {
      skip();
    }
  } // end while

  curToken += 1; // consume "}"
}

Token CellLibParser::getBooleanExpression() {
  // "..."
  curToken += 1; // consume "\""

  Token t = *curToken;
  auto prevToken = curToken;
  curToken += 1;
  for ( ; !isEndOfTokenStream(); prevToken = curToken, curToken += 1) {
    if ("\'" == *curToken || ")" == *curToken) {
      t += *curToken;
    }
    else if ("!" == *prevToken || "(" == *prevToken)  {
      t += *curToken;
    }
    else if ("\\" == *prevToken) {
      // similar to \"1A\"
      if ("\"" == *curToken) {
        do {
          t += *curToken;
          curToken += 1;
        } while (!isEndOfTokenStream() && "\"" != *curToken);
        t += *curToken; // append the last "\""
      }
      // similar to \a[1]
      else {
        do {
          t += *curToken;
          curToken += 1;
        } while (!isEndOfTokenStream() && "]" != *curToken);
        t += *curToken; // append "]"
      }
    }
    // end of quoted expression
    else if ("\"" == *curToken) {
      break;
    }
    // insert space between tokens
    else {
      t += " ";
      t += *curToken;   
    }
  } // end for

  curToken += 1; // consume "\""
  return t;
}

void CellLibParser::parseCellLeakagePower(Cell* cell) {
  // leakege_power () {...}
  curToken += 4; // consume "leakage_power", "(", ")" and "{"

  MyFloat value = 0.0;
  std::string when = "";
  while (!isEndOfGroup()) {
    // when: "...";
    if ("when" == *curToken) {
      curToken += 2; // consume "when" and ":"
      when = getBooleanExpression();
      curToken += 1; // consume ";"
    }
    // value: v;
    else if ("value" == *curToken) {
      curToken += 2; // consume "value" and ":"
      value = getMyFloat(*curToken);
      curToken += 2; // consume v and ";"
    }
    else {
      skip();
    }
  }

  cell->addLeakagePower(when, value);
  curToken += 1; // consume "}"
}

void CellLibParser::parseCellPin(Cell* cell) {
  // pin (name) {...}
  curToken += 2; // consume "pin" and "{"
  auto pin = cell->addCellPin(*curToken);
  curToken += 3; // consume name, ")" and "{"

  while (!isEndOfGroup()) {
    if ("timing" == *curToken) {
      parseTiming(pin->addTimingTable());
    }
    else if ("internal_power" == *curToken) {
      parseInternalPower(pin->addPowerTable());
    }
    // direction: value;
    else if ("direction" == *curToken) {
      curToken += 2; // consume "direction" and ":"
      if ("input" == *curToken) {
        pin->c[1] = lib->defaultInputPinCap;
        pin->c[0] = lib->defaultInputPinCap;
        cell->addInPin(pin);
      }
      else if("output" == *curToken) {
        pin->c[1] = lib->defaultOutputPinCap;
        pin->c[0] = lib->defaultOutputPinCap;
        cell->addOutPin(pin);
      }
      else if ("internal" == *curToken) {
        cell->addInternalPin(pin);
      }
      else if ("inout" == *curToken) {
        pin->c[1] = lib->defaultInoutPinCap;
        pin->c[0] = lib->defaultInoutPinCap;
        cell->addInOutPin(pin);
      }
      curToken += 2; // consume value and ";"
    }
    // fall_capacitance: value;
    else if ("fall_capacitance" == *curToken) {
      curToken += 2; // consume "fall_capacitance" and ":"
      pin->c[0] = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    // rise_capacitance: value;
    else if ("rise_capacitance" == *curToken) {
      curToken += 2; // consume "rise_capacitance" and ":"
      pin->c[1] = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    // max_capacitance: value;
    else if ("max_capacitance" == *curToken) {
      curToken += 2; // consume "max_capacitance" and ":"
      pin->maxC = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    // function: "...";
    else if ("function" == *curToken) {
      curToken += 2; // consume "function" and ":"
      pin->func = getBooleanExpression();
      curToken += 1; // consume ";"
    }
    // function_up: "...";
    else if ("function_up" == *curToken) {
      curToken += 2; // consume "function_up" and ":"
      pin->func_up = getBooleanExpression();
      curToken += 1; // consume ";"
    }
    // function_down: "...";
    else if ("function_down" == *curToken) {
      curToken += 2; // consume "function_down" and ":"
      pin->func_down = getBooleanExpression();
      curToken += 1; // consume ";"
    }
    // clock: true/false;
    else if ("clock" == *curToken) {
      curToken += 2; // consume "clock" and ":"
      if ("true" == *curToken) {
         cell->addClockPin(pin);
      }
      curToken += 2; // consume value and ";"
    }
    else {
      skip();
    }
  }

  curToken += 1; // consume "}"
}

void CellLibParser::parseCell() {
  // cell (name) {...}
  curToken += 2; // consume "cell" and "("
  auto cell = lib->addCell(*curToken);
  curToken += 3; // consume name, ")" and "{"

  while (!isEndOfGroup()) {
    if ("pin" == *curToken) {
      parseCellPin(cell);
    }
    else if ("leakage_power" == *curToken) {
      parseCellLeakagePower(cell);
    }
    // drive_strength: value;
    else if ("drive_strength" == *curToken) {
      curToken += 2; // consume "drive_strength" and ":"
      cell->driveStrength = std::stoul(*curToken);
      curToken += 2; // consume value and ";"
    }
    else if ("area" == *curToken) {
      curToken += 2; // consume "area" and ":"
      cell->area = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    else if ("cell_leakage_power" == *curToken) {
      curToken += 2; // consume "cell_leakage_power" and ":"
      cell->cellLeakagePower = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    else {
      skip();
    }
  }

  curToken += 1; // consume "}"
  cell->wrapUpConstruction();
}

static std::unordered_map<std::string, WireTreeType> mapName2WireTreeType = {
  {"best_case_tree",  BEST_CASE_TREE},
  {"balanced_tree",   BALANCED_TREE},
  {"worst_case_tree", WORST_CASE_TREE}
};

void CellLibParser::parseOperatingConditions() {
  // p[erating_conditions (name) {...}
  curToken += 2; // consume "operating_conditions" and "("
  lib->opCond = *curToken;
  curToken += 3; // consume name, ")" and "{"

  while(!isEndOfGroup()) {
    // tree_type = value;
    if ("tree_type" == *curToken) {
      curToken += 2; // consume "tree_type" and ":"
      lib->wireTreeType = mapName2WireTreeType.at(*curToken);
      curToken += 2; // consume value and ";"
    }
    else {
      skip();
    }
  }

  curToken += 1; // consume "}"
}

void CellLibParser::parseCellLibrary() {
  // library (libraryName) {...}
  curToken += 2; // consume "library" and "("
  lib->name = *curToken;
  curToken += 3; // consume lib->name, ")" "{"

  while (!isEndOfGroup()) {
    if ("wire_load" == *curToken) {
      parseWireLoad();
    }
    else if ("lu_table_template" == *curToken) {
      parseLutTemplate();
    }
    else if ("power_lut_template" == *curToken) {
      parseLutTemplate(true);
    }
    else if ("cell" == *curToken) {
      parseCell();
    }
    else if ("operating_conditions" == *curToken) {
      parseOperatingConditions();
    }
    // default_wire_load: "value";
    else if ("default_wire_load" == *curToken) {
      curToken += 3; // consume "default_wire_load", ":", and "\""
      lib->defaultWireLoad = lib->findWireLoad(*curToken);
      curToken += 3; // consume value, "\"", and ";"
    }
    // default_inout_pin_cap: value;
    else if ("default_inout_pin_cap" == *curToken) {
      curToken += 2; // consume "default_inout_pin_cap" and ":"
      lib->defaultInoutPinCap = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    // default_input_pin_cap: value;
    else if ("default_input_pin_cap" == *curToken) {
      curToken += 2; // consume "default_input_pin_cap" and ":"
      lib->defaultInputPinCap = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    // default_output_pin_cap: value;
    else if ("default_output_pin_cap" == *curToken) {
      curToken += 2; // consume "default_output_pin_cap" and ":"
      lib->defaultOutputPinCap = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    // default_max_transition: value;
    else if ("default_max_transition" == *curToken) {
      curToken += 2; // consume "default_max_transition" and ":"
      lib->defaultMaxSlew = getMyFloat(*curToken);
      curToken += 2; // consume value and ";"
    }
    else {
      skip();
    }
  }

  curToken += 1; // consume "}"
}

void CellLibParser::tokenizeFile(std::string inName) {
  std::vector<char> delimiters = {
    '(', ')', ',', ':', ';', '/', '#', '[', ']', '{', '}', 
    '*', '\"', '\\', '!', '^', '\'', '+', '&', '|'};
  std::vector<char> separators = {' ', '\t', '\n', '\r'};
  Tokenizer tokenizer(delimiters, separators);
  tokens = tokenizer.tokenize(inName); 
  curToken = tokens.begin();
}

void CellLibParser::parse(std::string inName) {
  tokenizeFile(inName);
  while (!isEndOfTokenStream()) {
    if ("library" == *curToken) {
      parseCellLibrary();
    }
    else {
      skip();
    }
  }
}

void CellLib::parse(std::string inName, bool toClear) {
  if (toClear) {
    clear();
    setup();
  }
  CellLibParser parser(this);
  parser.parse(inName);
}

static std::unordered_map<WireTreeType, std::string> mapWireTreeType2Name = {
  {BEST_CASE_TREE,  "best_case_tree"},
  {BALANCED_TREE,   "balanced_tree"},
  {WORST_CASE_TREE, "worst_case_tree"}
};

void CellLib::print(std::ostream& os) {
  os << "library (" << name << ") {" << std::endl;

  os << "  default_inout_pin_cap: " << defaultInoutPinCap << ";" << std::endl;
  os << "  default_input_pin_cap: " << defaultInputPinCap << ";" << std::endl;
  os << "  default_output_pin_cap: " << defaultOutputPinCap << ";" << std::endl;
  os << "  default_max_transition: " << defaultMaxSlew << ";" << std::endl;

  os << "  operating_conditions (" << opCond << ") {" << std::endl;
  os << "    tree_type: " << mapWireTreeType2Name.at(wireTreeType) << ";" << std::endl;
  os << "  }" << std::endl;

  for (auto& i: wireLoads) {
    i.second->print(os);
  }
  if (defaultWireLoad) {
    os << "  default_wire_load: \"" << defaultWireLoad->name << "\";" << std::endl;
  }

  for (auto& i: lutTemplates) {
    i.second->print(os);
  }

  for (auto& i: powerLutTemplates) {
    i.second->print(os);
  }

  for (auto& i: cells) {
    i.second->print(os);
  }

  os << "}" << std::endl;
}

void CellLib::clear() {
  for (auto& i: wireLoads) {
    delete i.second;
  }
  wireLoads.clear();

  for (auto& i: cells) {
    delete i.second;
  }
  cells.clear();

  for (auto& i: lutTemplates) {
    delete i.second;
  }
  lutTemplates.clear();

  for (auto& i: powerLutTemplates) {
    delete i.second;
  }
  powerLutTemplates.clear();
}

void CellLib::setup() {
  // add a LUT template for scalar case
  // default in liberty format
  auto scalar = addLutTemplate("scalar");
  scalar->shape.push_back(1);
  scalar->stride.push_back(1);

  // add a power LUT template for scalar case
  // default in liberty format
  scalar = addLutTemplate("scalar", true);
  scalar->shape.push_back(1);
  scalar->stride.push_back(1);
}

CellLib::CellLib() {
  setup();
}

CellLib::~CellLib() {
  clear();
}
