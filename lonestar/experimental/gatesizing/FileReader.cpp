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

#include "FileReader.h"

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

bool FileReader::isSeparator(char c) {
  for (size_t i = 0; i < numSeparators; i++) {
    if (c == separators[i]) {
      return true;
    }
  }
  return false;
}

bool FileReader::isDelimiter(char c) {
  for (size_t i = 0; i < numDelimiters; i++) {
    if (c == delimiters[i]) {
      return true;
    }
  }
  return false;
}

FileReader::FileReader(std::string inName, char* delimiters,
                       size_t numDelimiters, char* separators,
                       size_t numSeparators)
    : buffer(nullptr), bufferEnd(nullptr), cursor(nullptr),
      delimiters(delimiters), numDelimiters(numDelimiters),
      separators(separators), numSeparators(numSeparators) {
  std::ifstream ifs(inName);
  if (!ifs.is_open()) {
    std::cerr << "Cannot open " << inName << std::endl;
    std::abort();
  }

  size_t fileSize;
  ifs.seekg(0, std::ios::end);
  fileSize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  buffer = new char[fileSize + 1];
  ifs.read(buffer, fileSize);
  buffer[fileSize] = '\0';
  bufferEnd        = buffer + fileSize;
  cursor           = buffer;
  //  std::cout << buffer;
}

FileReader::~FileReader() { delete[] buffer; }

void FileReader::pushToken(std::string token) { tokenStack.push_back(token); }

std::string FileReader::nextToken() {
  std::string token;

  if (!tokenStack.empty()) {
    token = *(tokenStack.rbegin());
    tokenStack.pop_back();
  }

  else {
    for (; cursor < bufferEnd; ++cursor) {
      // skip a line of comment
      if (*cursor == '/' && *(cursor + 1) == '/') {
        while (*cursor != '\n') {
          ++cursor;
        }
      }

      // skip a block of comments
      if (*cursor == '/' && *(cursor + 1) == '*') {
        for (cursor = cursor + 2; cursor < bufferEnd; ++cursor) {
          if (*cursor == '*' && *(cursor + 1) == '/') {
            cursor += 2;
            break;
          }
        }
      }

      if (isSeparator(*cursor)) {
        if (!token.empty()) {
          ++cursor;
          break;
        }
      } else if (isDelimiter(*cursor)) {
        if (token.empty()) {
          token.push_back(*cursor);
          ++cursor;
        }
        break;
      } else {
        token.push_back(*cursor);
      }
    } // end for cursor
  }   // end else

  return token;
}
