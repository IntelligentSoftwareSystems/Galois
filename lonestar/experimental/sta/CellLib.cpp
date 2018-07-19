#include "CellLib.h"

void CellLibParser::skipAttribute() {
  std::cout << "Skip attribute " << *curToken << std::endl;
  while (curToken != tokens.end()) {
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
  while (curToken != tokens.end() && "{" != *curToken) {
    ++curToken;
  }
  ++curToken; // consume "{"

  // skip all statements in this group statement
  while (curToken != tokens.end() && "}" != *curToken) {
    skip();
  }
  ++curToken; // consume "}"
}

void CellLibParser::skip() {
  if (curToken == tokens.end()) {
    return;
  }

  // look ahead for statement types
  for (auto nextToken = curToken + 1; nextToken != tokens.end(); ++nextToken) {
    if (":" == *nextToken) {
      skipAttribute();
      break;
    }
    else if ("{" == *nextToken) {
      skipGroupStatement();
      break;
    }
    else if (";" == *nextToken) {
      skipAttribute();
      break;
    }
  }
}

void CellLibParser::parseCellLibrary() {
  while (curToken != tokens.end()) {
    // library (libraryName) {...}
    if ("library" == *curToken) {
      curToken += 2; // consume "("
      lib->name = *curToken;
      curToken += 3; // consume ")" and "{"

      // skip all statements until the end of library
      while ("}" != *curToken) {
        skip();
      }
      curToken += 1; // consume "}"
    }
    else {
      skip();
    }
  }
}

void CellLibParser::tokenizeFile(std::string inName) {
  std::vector<char> delimiters = {'(', ')', ',', ':', ';', '/', '#', '[', ']', '{', '}', '*', '\"', '\\'};
  std::vector<char> separators = {' ', '\t', '\n'};
  Tokenizer tokenizer(delimiters, separators);
  tokens = tokenizer.tokenize(inName); 
  curToken = tokens.begin();
}

void CellLibParser::parse(std::string inName) {
  tokenizeFile(inName);
  parseCellLibrary();
}

void CellLib::parse(std::string inName) {
  CellLibParser parser(this);
  parser.parse(inName);
}

void CellLib::print(std::ostream& os) {
  os << "library name: " << name << std::endl;
}
