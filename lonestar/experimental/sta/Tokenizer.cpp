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

#include "Tokenizer.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

bool Tokenizer::isSeparator(char c) {
  for (auto s: separators) {
    if (c == s) {
      return true;
    }
  }
  return false;
}

bool Tokenizer::isDelimiter(char c) {
  for (auto d: delimiters) {
    if (c == d) {
      return true;
    }
  }
  return false;
}

void Tokenizer::readFile2Buffer(std::string inName) {
  std::ifstream ifs(inName);
  if (!ifs.is_open()) {
    std::cerr << "Cannot open " << inName << std::endl;
    std::abort();
  }

  ifs.seekg(0, std::ios::end);
  size_t fileSize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  buffer = new char[fileSize + 1];
  ifs.read(buffer, fileSize);
  buffer[fileSize] = '\0';
  bufferEnd        = buffer + fileSize;
  cursor           = buffer;
  //  std::cout << buffer;
}

// assume comments in the form of {"form1_begin", "form1_end", ...}
// e.g., {"/*", "*/", "//", "\n"} for C-style comments
void Tokenizer::skipComment() {
  while (cursor < bufferEnd) {
    // check which form of comments is used
    auto cmtForm = comments.begin();
    auto endOfCmtForms = comments.end();
    for ( ; cmtForm != endOfCmtForms; cmtForm += 2) {
      bool isStartOfComment = true;
      auto& cmtStart = *cmtForm;
      for (size_t j = 0; j < cmtStart.size(); ++j) {
        if (cmtStart[j] != *(cursor+j)) {
          isStartOfComment = false;
          break;
        }
      }
      if (isStartOfComment) {
        break;
      }
    }

    // no comment begins at cursor
    if (endOfCmtForms == cmtForm) {
      return;
    }

    // remove the comment based on the form identified
    auto& cmtStart = *cmtForm;
    auto& cmtEnd = *(cmtForm+1);
    for (cursor += cmtStart.size(); cursor < bufferEnd; ++cursor) {
      bool isEndOfComment = true;
      for (size_t j = 0; j < cmtEnd.size(); ++j) {
        if (cmtEnd[j] != *(cursor+j)) {
          isEndOfComment = false;
          break;
        }
      }
      if (isEndOfComment) {
        cursor += cmtEnd.size();
        break;
      }
    }
  } // end while
}

Token Tokenizer::getNextToken() {
  Token t;
  for ( ; cursor < bufferEnd; ++cursor) {
    skipComment();

    // a separator marks the end of token
    if (isSeparator(*cursor)) {
      if (!t.empty()) {
        ++cursor;
        break;
      }
    }

    // a delimiter is itself a token
    else if (isDelimiter(*cursor)) {
      if (t.empty()) {
        t.push_back(*cursor);
        ++cursor;
      }
      break;
    }

    // in the middle of a token
    else {
      t.push_back(*cursor);
    }
  } // end for cursor

  return t;
}

std::vector<Token> Tokenizer::tokenize(std::string inName) {
  readFile2Buffer(inName);

  std::vector<Token> tokens;
  Token t = getNextToken();
  while (!t.empty()) {
    tokens.push_back(t);
    t = getNextToken();
  }

  delete[] buffer;
  return tokens;
}
