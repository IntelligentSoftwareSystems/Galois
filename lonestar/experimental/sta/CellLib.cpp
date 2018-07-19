#include "CellLib.h"

void LutTemplate::print(std::ostream& os) {
  os << "lu_table_template (" << name << ") {" << std::endl;
  size_t i = 0;
  for (auto& v: var) {
    os << "  variable_" << ++i << ": " << v << std::endl;
  }

  i = 0;
  for (auto& d: dim) {
    os << "  index_" << ++i << " (\"";
    size_t j = 1;
    os << float(j) * 0.0010f;
    while (j < d) {
      os << "," << (float)(++j) * 0.0010f;
    }
    os << "\");" << std::endl;
  }
  os << "}" << std::endl;
}

float WireLoad::wireR(size_t deg) {
  return 0.0;
}

float WireLoad::wireC(size_t deg) {
  return 0.0;
}

void WireLoad::print(std::ostream& os) {
  os << "wire_load (\"" << name << "\") {" << std::endl;
  os << "  capacitance: " << c << ";" << std::endl;
  os << "  resistance: " << r << ";" << std::endl;
  os << "  slope: " << slope << ";" << std::endl;
  for (auto& i: fanoutLength) {
    os << "  fanout_length ( " << i.first << ", " << i.second << " );" << std::endl;
  }
  os << "}" << std::endl;
}

void CellLibParser::skipAttribute() {
  std::cout << "Skip attribute " << *curToken << std::endl;
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

void CellLibParser::skipGroupStatement() {
  std::cout << "Skip group statement " << *curToken << std::endl;
  while (!isEndOfTokenStream() && "{" != *curToken) {
    ++curToken;
  }
  ++curToken; // consume "{"

  // skip all statements in this group statement
  while (!isEndOfGroup()) {
    skip();
  }
  ++curToken; // consume "}"
}

void CellLibParser::skip() {
  if (isEndOfTokenStream()) {
    return;
  }

  // look ahead for statement types
  bool skipped = false;
  for (auto nextToken = curToken + 1; nextToken != tokens.end(); ++nextToken) {
    // simple attributes
    // attr_key_word: value;
    if (":" == *nextToken) {
      skipAttribute();
      skipped = true;
      break;
    }
    // group statements
    // group_key_word (...) {...}
    else if ("{" == *nextToken) {
      skipGroupStatement();
      skipped = true;
      break;
    }
    // complex attributes
    // attr_key_word (...) {...}
    else if (";" == *nextToken) {
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
  // wire_load (\"name\") {...}
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

  size_t unmatched = 0;
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

void CellLibParser::parseLut() {
  skip();
}

void CellLibParser::parseCellPin() {
  skip();
}

void CellLibParser::parseCell() {
  skip();
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
    else {
//      std::cout << "Skip statement " << *curToken << std::endl;
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
  os << "library name: " << name << std::endl;

  for (auto& i: wireLoads) {
    i.second->print(os);
  }
  os << "default_wire_load: \"" << defaultWireLoad->name << "\";" << std::endl;

  for (auto& i: lutTemplates) {
    i.second->print(os);
  }
}

void CellLib::clear() {
  for (auto& i: wireLoads) {
    delete i.second;
  }

  for (auto& i: lutTemplates) {
    delete i.second;
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
