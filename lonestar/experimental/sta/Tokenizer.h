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

#include <string>
#include <vector>

#ifndef GALOIS_EDA_TOKENIZER_H
#define GALOIS_EDA_TOKENIZER_H

using Token = std::string;

class Tokenizer {
  std::vector<char>& delimiters;
  std::vector<char>& separators;
  char* buffer;
  char* bufferEnd;
  char* cursor;

private:
  bool isSeparator(char c);
  bool isDelimiter(char c);
  void readFile2Buffer(std::string inName);
  void skipComment();
  Token getNextToken();

public:
  Tokenizer(std::vector<char>& d, std::vector<char>& s): delimiters(d), separators(s) {}
  std::vector<Token> tokenize(std::string inName);
};

#endif // GALOIS_EDA_TOKENIZER_H
