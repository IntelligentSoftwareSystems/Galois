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

bool CellPin::isEdgeDefined(CellPin* inPin, bool isNeg, bool isRise, TableType index) {
  return !tables[inPin][isRise][index][isNeg].empty();
}

MyFloat CellPin::extract(Parameter& param, TableType index, CellPin* inPin, bool isNeg, bool isRise, std::string when) {
  return tables[inPin][isRise][index][isNeg][when]->lookup(param);
}

std::pair<MyFloat, std::string>
CellPin::extractMax(Parameter& param, TableType index, CellPin* inPin, bool isNeg, bool isRise) {
  MyFloat ret = -std::numeric_limits<MyFloat>::infinity();
  std::string when;
  for (auto& i: tables[inPin][isRise][index][isNeg]) {
    auto tmp = i.second->lookup(param);
    if (tmp > ret) {
      ret = tmp;
      when = i.first;
    }
  }
  return {ret, when};
}

std::pair<MyFloat, std::string>
CellPin::extractMin(Parameter& param, TableType index, CellPin* inPin, bool isNeg, bool isRise) {
  MyFloat ret = std::numeric_limits<MyFloat>::infinity();
  std::string when;
  for (auto& i: tables[inPin][isRise][index][isNeg]) {
    auto tmp = i.second->lookup(param);
    if (tmp < ret) {
      ret = tmp;
      when = i.first;
    }
  }
  return {ret, when};
}

void CellPin::print(std::ostream& os) {
  os << "    pin (" << name << ") {" << std::endl;

  std::string d = (isInput && !isOutput) ? "input" :
                  (!isInput && isOutput) ? "output" :
                  (isInput && isOutput) ? "inout" : "internal";
  os << "      direction: " << d << ";" << std::endl;

  if (isInput) {
    os << "      capactance: " << ((c[1] > c[0]) ? c[1] : c[0]) << ";" << std::endl;
    os << "      rise_capacitance: " << c[1] << ";" << std::endl;
    os << "      fall_capacitance: " << c[0] << ";" << std::endl;
  }

  if (isOutput) {
    os << "      max_capacitance: " << maxC << ";" << std::endl;
    os << "      function: \"" << func << "\";" << std::endl;

    // convert tables to printing order
    // order of keys: pin, unateness, when, delay/slew, fall/rise
    using InnerMap = std::unordered_map<std::string, Lut*[2][2]>;
    using OuterMap = std::unordered_map<CellPin*, InnerMap[2]>;
    OuterMap printTables;
    for (auto& i: tables) {
      auto pin = i.first;
      for (int fr = 0; fr < 2; fr++) {
        for (int ds = 0; ds < 2; ds++) {
          for (int pn = 0; pn < 2; pn++) {
            for (auto& j: i.second[fr][ds][pn]) {
              printTables[pin][pn][j.first][ds][fr] = j.second;
            }
          }
        }
      }
    }

    // print tables
    for (auto& i: printTables) {
      auto pin = i.first;
      auto outMap = i.second;
      for (int pn = 0; pn < 2; pn++) {
        for (auto& j: outMap[pn]) {
          auto& when = j.first;
          std::string unateness;
          if (0 == pn) {
            unateness = (outMap[1].count(when)) ? "positive_unate" : "non_unate";
          }
          else {
            unateness = "negative_unate";
            if (outMap[0].count(when)) {
              continue;
            }
          }

          os << "      timing () {" << std::endl;
          os << "        related_pin: \"" << pin->name << "\";" << std::endl;
          if (!when.empty()) {
            os << "        when: \"" << when << "\";" << std::endl;
          }
          os << "        timing_sense: " << unateness << ";" << std::endl;

          auto& t = j.second;
          auto lut = t[TABLE_DELAY][0];
          if (lut) {
            lut->print("cell_fall", os);
          }
          lut = t[TABLE_DELAY][1];
          if (lut) {
            lut->print("cell_rise", os);
          }
          lut = t[TABLE_SLEW][0];
          if (lut) {
            lut->print("fall_transition", os);
          }
          lut = t[TABLE_SLEW][1];
          if (lut) {
            lut->print("rise_transition", os);
          }
          os << "      }" << std::endl;
        }
      }
    } // end for printTables
  } // end if (isOutput)

  os << "    }" << std::endl;
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
    i.second->print(os);
  }

  for (auto& i: internalPins) {
    i.second->print(os);
  }

  for (auto& i: outPins) {
    i.second->print(os);
  }

  os << "  }" << std::endl;
}

void LutTemplate::print(std::ostream& os) {
  os << "  lu_table_template (" << name << ") {" << std::endl;
  size_t i = 0;
  for (auto& v: var) {
    os << "    variable_" << ++i << ": " << v << ";" << std::endl;
  }

  i = 0;
  for (auto& d: shape) {
    os << "    index_" << ++i << " (\"";
    size_t j = 1;
    os << MyFloat(j) * 0.0010f;
    while (j < d) {
      os << ", " << (MyFloat)(++j) * 0.0010f;
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
  if (TREE_TYPE_BEST_CASE == lib->wireTreeType) {
    return 0.0;
  }
  else {
    auto deg = wire->outDeg();
    auto wL = wireLength(deg);
    auto wC = c * wL;
    auto wR = r * wL;

    // balanced tree
    if (TREE_TYPE_BALANCED == lib->wireTreeType) {
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

static std::unordered_map<std::string, VariableType> mapVarName2Type = {
  {"input_transition_time",        VARIABLE_INPUT_TRANSITION_TIME},
  {"constrained_pin_transition",   VARIABLE_CONSTRAINED_PIN_TRANSITION},
  {"related_pin_transition",       VARIABLE_RELATED_PIN_TRANSITION},
  {"total_output_net_capacitance", VARIABLE_TOTAL_OUTPUT_NET_CAPACITANCE},
  {"input_net_transition",         VARIABLE_INPUT_NET_TRANSITION},
  {"time",                         VARIABLE_TIME}
};

void CellLibParser::parseLutTemplate() {
  // lu_table_template (name) {...}
  curToken += 2; // consume "lu_table_tamplate" and "("
  auto lutTemplate = lib->addLutTemplate(*curToken);
  curToken += 3; // consume name, ")", and "{"

  int unmatched = 0;
  while (!isEndOfGroup()) {
    // variable_*: name;
    if ("variable_1" == *curToken || "variable_2" == *curToken || "variable_3" == *curToken) {
      curToken += 2; // consume "variable_*" and ":"
      lutTemplate->var.push_back(mapVarName2Type.at(*curToken));
      curToken += 2; // consume name and ";"
      unmatched++;
    }
    // index_* ("num1[,num*]");
    else if ("index_1" == *curToken || "index_2" == *curToken || "index_3" == *curToken) {
      curToken += 4; // consume "index_*", "(", "\"" and num1
      size_t num = 1;
      while ("," == *curToken) {
        curToken += 2; // consume "," and num*
        num++;
      }
      lutTemplate->shape.push_back(num);
      curToken += 3; // consume "\"", ")" and ";"
      unmatched--;
    }
    else {
      skip();
    }
  }

  // find stride for each dimension
  auto& shape = lutTemplate->shape;
  auto& stride = lutTemplate->stride;
  stride.insert(stride.begin(), shape.size(), 1);
  for (size_t i = shape.size() - 1; i >= 1; --i) {
    stride[i-1] = stride[i] * shape[i];
  }

  assert(!unmatched);
  curToken += 1; // consume "}"
}

void CellLibParser::parseLut(Lut* lut) {
  // key_word (name) {...}
  curToken += 2; // consume key_word and "("
  lut->lutTemplate = lib->findLutTemplate(*curToken);
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
}

void CellLibParser::parseTiming(CellPin* pin) {
  // timing () {...}
  curToken += 4; // consume "timing", "(", ")" and "{"

  bool isPos = false, isNeg = false;
  CellPin* relatedPin = nullptr;
  Token when = "";
  while (!isEndOfGroup()) {
    if ("cell_fall" == *curToken) {
      if (!isPos && !isNeg) {
        skip();
        continue;
      }
      auto lut = lib->addLut();
      pin->addLut(lut, TABLE_DELAY, false, relatedPin, when, isPos, isNeg);
      parseLut(lut);
    }
    else if ("cell_rise" == *curToken) {
      if (!isPos && !isNeg) {
        skip();
        continue;
      }
      auto lut = lib->addLut();
      pin->addLut(lut, TABLE_DELAY, true, relatedPin, when, isPos, isNeg);
      parseLut(lut);
    }
    else if ("fall_transition" == *curToken) {
      if (!isPos && !isNeg) {
        skip();
        continue;
      }
      auto lut = lib->addLut();
      pin->addLut(lut, TABLE_SLEW, false, relatedPin, when, isPos, isNeg);
      parseLut(lut);
    }
    else if ("rise_transition" == *curToken) {
      if (!isPos && !isNeg) {
        skip();
        continue;
      }
      auto lut = lib->addLut();
      pin->addLut(lut, TABLE_SLEW, true, relatedPin, when, isPos, isNeg);
      parseLut(lut);
    }
    // when: "...";
    else if ("when" == *curToken) {
      curToken += 2; // consume "when" and ":"
      when = getBooleanExpression();
      curToken += 1; // consume ";"
    }
    // related_pin: "name";
    else if ("related_pin" == *curToken) {
      curToken += 3; // consume "related_pin", ":" and "\""
      relatedPin = pin->cell->findCellPin(*curToken);
      curToken += 3; // consume name, "\"" and ";"
    }
    // timing_sense: value;
    else if ("timing_sense" == *curToken) {
      curToken += 2; // consume "timing_sense" and ":"
      if ("positive_unate" == *curToken) {
        isPos = true;
      }
      else if ("negative_unate" == *curToken) {
        isNeg = true;
      }
      else if ("non_unate" == *curToken) {
        isPos = true;
        isNeg = true;
      }
      curToken += 2; // consume value and ";"
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
    if ("timing" == *curToken && pin->isOutput) {
      parseTiming(pin);
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
}

void CellLibParser::parseOperatingConditions() {
  // p[erating_conditions (name) {...}
  curToken += 2; // consume "operating_conditions" and "("
  lib->opCond = *curToken;
  curToken += 3; // consume name, ")" and "{"

  while(!isEndOfGroup()) {
    // tree_type = value;
    if ("tree_type" == *curToken) {
      curToken += 2; // consume "tree_type" and ":"
      if ("best_case_tree" == *curToken) {
        lib->wireTreeType = TREE_TYPE_BEST_CASE;
      }
      else if ("balanced_tree" == *curToken) {
        lib->wireTreeType = TREE_TYPE_BALANCED;
      }
      else if ("worst_case_tree" == *curToken) {
        lib->wireTreeType = TREE_TYPE_WORST_CASE;
      }
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

void CellLib::print(std::ostream& os) {
  os << "library (" << name << ") {" << std::endl;

  os << "  default_inout_pin_cap: " << defaultInoutPinCap << ";" << std::endl;
  os << "  default_input_pin_cap: " << defaultInputPinCap << ";" << std::endl;
  os << "  default_output_pin_cap: " << defaultOutputPinCap << ";" << std::endl;
  os << "  default_max_transition: " << defaultMaxSlew << ";" << std::endl;

  os << "  operating_conditions (" << opCond << ") {" << std::endl;
  std::string treeTypeName[] = {"best_case_tree", "balanced_tree", "worst_case_tree"};
  os << "    tree_type: " << treeTypeName[wireTreeType] << ";" << std::endl;
  os << "  }" << std::endl;

  for (auto& i: wireLoads) {
    i.second->print(os);
  }
  os << "  default_wire_load: \"" << defaultWireLoad->name << "\";" << std::endl;

  for (auto& i: lutTemplates) {
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
    auto c = i.second;
    for (auto& j: c->pins) {
      delete j.second;
    }
    c->pins.clear();
    delete c;
  }
  cells.clear();

  for (auto& i: lutTemplates) {
    delete i.second;
  }
  lutTemplates.clear();

  for (auto& i: luts) {
    delete i;
  }
  luts.clear();
}

void CellLib::setup() {
  // add a LUT template for scalar case
  auto scalar = addLutTemplate("scalar");
  scalar->shape.push_back(1);
  scalar->stride.push_back(1);
}

CellLib::CellLib() {
  setup();
}

CellLib::~CellLib() {
  clear();
}
