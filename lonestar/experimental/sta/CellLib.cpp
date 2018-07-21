#include <cassert>

#include "CellLib.h"

float Lut::lookup(std::vector<float>& param) {
  return 0.0;
}

void Lut::print(std::string attr, std::ostream& os) {
  os << "        " << attr << "(" << lutTemplate->name << ") {" << std::endl;
  os << "";
  os << "        }" << std::endl;
}

bool CellPin::isUnateAtEdge(CellPin* inPin, bool isNeg, bool isRise) {
  return true;
}

float CellPin::extract(TableType index, CellPin* inPin, bool isNeg, bool isRise, std::string when) {
  return 0.0;
}

std::pair<float, std::string>
CellPin::extractMaxDelay(CellPin* inPin, bool isNeg, bool isRise) {
  return {0.0, ""};
}

void CellPin::print(std::ostream& os) {
  os << "    pin (" << name << ") {" << std::endl;

  std::string d = (isInput && !isOutput) ? "input" :
                  (!isInput && isOutput) ? "output" :
                  (isInput && isOutput) ? "inout" : "internal";
  os << "      direction: " << d << ";" << std::endl;

  if (isInput) {
    os << "      capactance: " << ((riseC > fallC) ? riseC : fallC) << ";" << std::endl;
    os << "      rise_capacitance: " << riseC << ";" << std::endl;
    os << "      fall_capacitance: " << fallC << ";" << std::endl;
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
  for (auto& d: dim) {
    os << "    index_" << ++i << " (\"";
    size_t j = 1;
    os << float(j) * 0.0010f;
    while (j < d) {
      os << "," << (float)(++j) * 0.0010f;
    }
    os << "\");" << std::endl;
  }
  os << "  }" << std::endl;
}

float WireLoad::wireR(size_t deg) {
  return 0.0;
}

float WireLoad::wireC(size_t deg) {
  return 0.0;
}

void WireLoad::print(std::ostream& os) {
  os << "  wire_load (\"" << name << "\") {" << std::endl;
  os << "    capacitance: " << c << ";" << std::endl;
  os << "    resistance: " << r << ";" << std::endl;
  os << "    slope: " << slope << ";" << std::endl;
  for (auto& i: fanoutLength) {
    os << "    fanout_length (" << i.first << "," << i.second << ");" << std::endl;
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
      if (isTopCall) {
        std::cout << "Skip attribute " << *curToken << std::endl;
      }
      skipAttribute();
      skipped = true;
      break;
    }
    // group statements
    // group_key_word (...) {...}
    else if ("{" == *nextToken) {
      if (isTopCall) {
        std::cout << "Skip group statement " << *curToken << std::endl;
      }
      skipGroup();
      skipped = true;
      break;
    }
    // complex attributes
    // attr_key_word (...);
    else if (";" == *nextToken) {
      if (isTopCall) {
        std::cout << "Skip attribute " << *curToken << std::endl;
      }
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
  WireLoad* wireLoad = new WireLoad();
  wireLoad->name = *curToken;
  lib->wireLoads[wireLoad->name] = wireLoad;
  curToken += 4; // consume name, "\"", ")", and "{"

  while (!isEndOfGroup()) {
    // capacitance: value;
    if ("capacitance" == *curToken) {
      curToken += 2; // consume "capacitance" and ":"
      wireLoad->c = std::stof(*curToken);
      curToken += 2; // consume value and ";"
    }
    // resisance: value;
    else if ("resistance" == *curToken) {
      curToken += 2; // consume "resistance" and ":"
      wireLoad->r = std::stof(*curToken);
      curToken += 2; // consume value and ";"
    }
    // slope: value;
    else if ("slope" == *curToken) {
      curToken += 2; // consume "slope" and ":"
      wireLoad->slope = std::stof(*curToken);
      curToken += 2; // consume value and ";"
    }
    // fanout_length(fanout, length);
    else if ("fanout_length" == *curToken) {
      curToken += 2; // consume "fanout_length" and "("
      size_t fanout = std::stoul(*curToken);
      curToken += 2; // consume fanout and ","
      float length = std::stoul(*curToken);
      wireLoad->fanoutLength[fanout] = length;
      curToken += 3; // consume length, ")" and ";"
    }
    else {
      skip();
    }
  }

  curToken += 1; // consume "}"
}

void CellLibParser::parseLutTemplate() {
  // lu_table_template (name) {...}
  curToken += 2; // consume "lu_table_tamplate" and "("
  LutTemplate* lutTemplate = new LutTemplate;
  lutTemplate->name = *curToken;
  lib->lutTemplates[lutTemplate->name] = lutTemplate;
  curToken += 3; // consume name, ")", and "{"

  int unmatched = 0;
  while (!isEndOfGroup()) {
    // variable_*: name;
    if ("variable_1" == *curToken || "variable_2" == *curToken || "variable_3" == *curToken) {
      curToken += 2; // consume "variable_*" and ":"
      lutTemplate->var.push_back(*curToken);
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
      lutTemplate->dim.push_back(num);
      curToken += 3; // consume "\"", ")" and ";"
      unmatched--;
    }
    else {
      skip();
    }
  }
  
  assert(!unmatched);
  curToken += 1; // consume "}"
}

void CellLibParser::parseLut(Lut* lut) {
  // key_word (name) {...}
  curToken += 2; // consume key_word and "("
  lut->lutTemplate = lib->lutTemplates[*curToken];
  curToken += 3; // consume name, ")" and "{"

  while (!isEndOfGroup()) {
    // index_* ("num1[,num*]");
    if ("index_1" == *curToken || "index_2" == *curToken || "index_3" == *curToken) {
      curToken += 3; // consume "index_*", "(" and "\""
      std::vector<float> v;
      while (!isEndOfTokenStream() && ")" != *curToken) {
        v.push_back(std::stof(*curToken));
        curToken += 2; // consume num*, and "," in between or "\"" at the end
      }
      lut->index.push_back(v);
      curToken += 2; // consume ")" and ";"
    }
    // values ("num1[,num*]"[, \ "num1[,num*]"]);
    else if ("values" == *curToken) {
      curToken += 3; // consume "value", "(" and "\""
      while (!isEndOfTokenStream() && ")" != *curToken) {
        lut->value.push_back(std::stof(*curToken));
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
      Lut* lut = new Lut;
      lib->luts.insert(lut);
      if (isPos) {
        pin->tables[relatedPin][0][TABLE_DELAY][0][when] = lut;
      }
      if (isNeg) {
        pin->tables[relatedPin][0][TABLE_DELAY][1][when] = lut;
      }
      parseLut(lut);
    }
    else if ("cell_rise" == *curToken) {
      if (!isPos && !isNeg) {
        skip();
        continue;
      }
      Lut* lut = new Lut;
      lib->luts.insert(lut);
      if (isPos) {
        pin->tables[relatedPin][1][TABLE_DELAY][0][when] = lut;
      }
      if (isNeg) {
        pin->tables[relatedPin][1][TABLE_DELAY][1][when] = lut;
      }
      parseLut(lut);
    }
    else if ("fall_transition" == *curToken) {
      if (!isPos && !isNeg) {
        skip();
        continue;
      }
      Lut* lut = new Lut;
      lib->luts.insert(lut);
      if (isPos) {
        pin->tables[relatedPin][0][TABLE_SLEW][0][when] = lut;
      }
      if (isNeg) {
        pin->tables[relatedPin][0][TABLE_SLEW][1][when] = lut;
      }
      parseLut(lut);
    }
    else if ("rise_transition" == *curToken) {
      if (!isPos && !isNeg) {
        skip();
        continue;
      }
      Lut* lut = new Lut;
      lib->luts.insert(lut);
      if (isPos) {
        pin->tables[relatedPin][1][TABLE_SLEW][0][when] = lut;
      }
      if (isNeg) {
        pin->tables[relatedPin][1][TABLE_SLEW][1][when] = lut;
      }
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
      relatedPin = pin->cell->inPins[*curToken];
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

  float value = 0.0;
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
      value = std::stof(*curToken);
      curToken += 2; // consume v and ";"
    }
    else {
      skip();
    }
  }

  cell->leakagePower[when] = value;
  curToken += 1; // consume "}"
}

void CellLibParser::parseCellPin(Cell* cell) {
  // pin (name) {...}
  curToken += 2; // consume "pin" and "{"
  CellPin* pin = new CellPin;
  pin->name = *curToken;
  pin->cell = cell;
  cell->pins[pin->name] = pin;
  curToken += 3; // consume name, ")" and "{"

  while (!isEndOfGroup()) {
    if ("timing" == *curToken && pin->isOutput) {
      parseTiming(pin);
    }
    // direction: value;
    else if ("direction" == *curToken) {
      curToken += 2; // consume "direction" and ":"
      if ("input" == *curToken) {
        pin->isInput = true;
        pin->isOutput = false;
        pin->riseC = lib->defaultInputPinCap;
        pin->fallC = lib->defaultInputPinCap;
        cell->inPins[pin->name] = pin;
      }
      else if("output" == *curToken) {
        pin->isInput = false;
        pin->isOutput = true;
        pin->riseC = lib->defaultOutputPinCap;
        pin->fallC = lib->defaultOutputPinCap;
        cell->outPins[pin->name] = pin;
     }
      else if ("internal" == *curToken) {
        pin->isInput = false;
        pin->isOutput = false;
        cell->internalPins[pin->name] = pin;
      }
      else if ("inout" == *curToken) {
        pin->isInput = true;
        pin->isOutput = true;
        pin->riseC = lib->defaultInoutPinCap;
        pin->fallC = lib->defaultInoutPinCap;
        cell->inPins[pin->name] = pin;
        cell->outPins[pin->name] = pin;
      }
      curToken += 2; // consume value and ";"
    }
    // fall_capacitance: value;
    else if ("fall_capacitance" == *curToken) {
      curToken += 2; // consume "fall_capacitance" and ":"
      pin->fallC = std::stof(*curToken);
      curToken += 2; // consume value and ";"
    }
    // rise_capacitance: value;
    else if ("rise_capacitance" == *curToken) {
      curToken += 2; // consume "rise_capacitance" and ":"
      pin->riseC = std::stof(*curToken);
      curToken += 2; // consume value and ";"
    }
    // max_capacitance: value;
    else if ("max_capacitance" == *curToken) {
      curToken += 2; // consume "max_capacitance" and ":"
      pin->maxC = std::stof(*curToken);
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
  Cell* cell = new Cell;
  cell->name = *curToken;
  lib->cells[cell->name] = cell;
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
      cell->area = std::stof(*curToken);
      curToken += 2; // consume value and ";"
    }
    else if ("cell_leakage_power" == *curToken) {
      curToken += 2; // consume "cell_leakage_power" and ":"
      cell->cellLeakagePower = std::stof(*curToken);
      curToken += 2; // consume value and ";"
    }
    else {
      skip();
    }
  }

  curToken += 1; // consume "}"
}

void CellLibParser::parseOperatingCondition() {
  skip();
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
    else if ("operating_condition" == *curToken) {
      parseOperatingCondition();
    }
    // default_wire_load: "value";
    else if ("default_wire_load" == *curToken) {
      curToken += 3; // consume "default_wire_load", ":", and "\""
      lib->defaultWireLoad = lib->wireLoads[*curToken];
      curToken += 3; // consume value, "\"", and ";"
    }
    // default_inout_pin_cap: value;
    else if ("default_inout_pin_cap" == *curToken) {
      curToken += 2; // consume "default_inout_pin_cap" and ":"
      lib->defaultInoutPinCap = std::stof(*curToken);
      curToken += 2; // consume value and ";"
    }
    // default_input_pin_cap: value;
    else if ("default_input_pin_cap" == *curToken) {
      curToken += 2; // consume "default_input_pin_cap" and ":"
      lib->defaultInputPinCap = std::stof(*curToken);
      curToken += 2; // consume value and ";"
    }
    // default_output_pin_cap: value;
    else if ("default_output_pin_cap" == *curToken) {
      curToken += 2; // consume "default_output_pin_cap" and ":"
      lib->defaultOutputPinCap = std::stof(*curToken);
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

  for (auto& i: cells) {
    auto c = i.second;
    for (auto& j: c->pins) {
      delete j.second;
    }
    delete c;
  }

  for (auto& i: lutTemplates) {
    delete i.second;
  }

  for (auto& i: luts) {
    delete i;
  }
}

void CellLib::setup() {
  // add a LUT template for scalar case
  LutTemplate* scalar = new LutTemplate;
  scalar->name = "scalar";
  scalar->var.push_back("");
  scalar->dim.push_back(1);
  lutTemplates[scalar->name] = scalar; 
}

CellLib::CellLib() {
  setup();
}

CellLib::~CellLib() {
  clear();
}
