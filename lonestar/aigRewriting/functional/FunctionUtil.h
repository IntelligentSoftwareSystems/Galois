/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

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
