/*
 * FunctionUtil.h
 *
 *  Created on: 14/02/2017
 *      Author: possani
 */

#ifndef SRC_FUNCTION_FUNCTIONUTIL_H_
#define SRC_FUNCTION_FUNCTIONUTIL_H_

#include "BitVectorPool.h"

#include <cstdlib>
#include <string>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <vector>
#include <unordered_map>

namespace Functional {

enum Token {
  ANDop = '*',
  ORop  = '+',
  XORop = '^',
  LP    = '(',
  RP    = ')',
  NOTop = '!',
  LIT,
  END = ';',
  EMPTY
};

typedef unsigned long int word;
typedef std::unordered_map<std::string, std::pair<word*, unsigned int>>
    StringFunctionMap;

class FunctionUtil {

  Token currentToken;
  std::string tokenValue;
  StringFunctionMap& literals;
  BitVectorPool& functionPool;
  int nVars;
  int nWords;

public:
  FunctionUtil(StringFunctionMap& entries, BitVectorPool& functionPool,
               int nVars, int nWords);

  virtual ~FunctionUtil();

  word* parseExpression(std::string expression);
  word* prim(std::istringstream& expression);
  word* term(std::istringstream& expression);
  word* expr2(std::istringstream& expression);
  word* expr1(std::istringstream& expression);
  Token getToken(std::istringstream& expression);
  word* parseHexa(std::string hexa);
};

} // namespace Functional

#endif /* SRC_FUNCTION_FUNCTIONUTIL_H_ */
