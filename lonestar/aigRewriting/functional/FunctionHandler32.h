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

/**
 * This class represents the Function basic data structure. A Function is a
 * vector of unsigned long integers that represents the truth table of a
 * boolean function.
 *
 * @author Marcos Henrique Backes - mhbackes@inf.ufrgs.br.
 *
 * @see InputNode, ChoiceNode.
 *
 * Modified by Vinicius Possani
 * Last modification in July 28, 2017.
 */

#ifndef FUNCTIONAL32_H_
#define FUNCTIONAL32_H_

#include "../xxHash/xxhash.h"
#include <cmath>
#include <cassert>
#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace Functional32 {

typedef unsigned int word;

static inline void copy(word* result, word* original, int nWords);
static inline void NOT(word* result, word* original, int nWords);
static inline void AND(word* result, word* lhs, word* rhs, int nWords);
static inline void NAND(word* result, word* lhs, word* rhs, int nWords);
static inline void OR(word* result, word* lhs, word* rhs, int nWords);
static inline void XOR(word* result, word* lhs, word* rhs, int nWords);
static inline bool isConstZero(word* function, int nVars);
static inline bool isConstOne(word* function, int nVars);
static inline int countOnes(unsigned uWord);
static inline int wordNum(int nVars);
static void truthStretch(word* result, word* input, int inVars, int nVars,
                         unsigned phase);
static void swapAdjacentVars(word* result, word* input, int nVars, int iVar);
static std::string toCubeString( word* function, int nWords, int nVars);
static std::string toHex(word* function, int nWords);
static std::string toBin(word* function, int nWords);

void copy(word* result, word* original, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = original[i];
  }
}

void NOT(word* result, word* original, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = ~(original[i]);
  }
}

void AND(word* result, word* lhs, word* rhs, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = lhs[i] & rhs[i];
  }
}

void NAND(word* result, word* lhs, word* rhs, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = ~(lhs[i] & rhs[i]);
  }
}

void OR(word* result, word* lhs, word* rhs, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = lhs[i] | rhs[i];
  }
}

void XOR(word* result, word* lhs, word* rhs, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = lhs[i] ^ rhs[i];
  }
}

bool isConstZero(word* function, int nVars) {

  word* pFunction = function;
  word* pLimit    = pFunction + wordNum(nVars);

  while (pFunction != pLimit) {
    if (*pFunction++ != 0U) {
      return false;
    }
  }

  return true;
}

bool isConstOne(word* function, int nVars) {

  word* pFunction = function;
  word* pLimit    = pFunction + wordNum(nVars);
  const word ONE  = ~0U;

  while (pFunction != pLimit) {
    if (*pFunction++ != ONE) {
      return false;
    }
  }

  return true;
}

int countOnes(unsigned uWord) {

    uWord = (uWord & 0x55555555) + ((uWord>>1) & 0x55555555);
    uWord = (uWord & 0x33333333) + ((uWord>>2) & 0x33333333);
    uWord = (uWord & 0x0F0F0F0F) + ((uWord>>4) & 0x0F0F0F0F);
    uWord = (uWord & 0x00FF00FF) + ((uWord>>8) & 0x00FF00FF);
    return  (uWord & 0x0000FFFF) + (uWord>>16);
}

int wordNum(int nVars) { return nVars <= 5 ? 1 : 1 << (nVars - 5); }

inline void truthStretch(word* result, word* input, int inVars, int nVars,
                  unsigned phase) {

  unsigned* pTemp;
  int var = inVars - 1, counter = 0;

  for (int i = nVars - 1; i >= 0; i--) {

    if (phase & (1 << i)) {

      for (int j = var; j < i; j++) {

        swapAdjacentVars(result, input, nVars, j);
        pTemp  = input;
        input  = result;
        result = pTemp;
        counter++;
      }
      var--;
    }
  }

  assert(var == -1);

  // swap if it was moved an even number of times
  int nWords = wordNum(nVars);
  if (!(counter & 1)) {
    copy(result, input, nWords);
  }
}

void swapAdjacentVars(word* result, word* input, int nVars, int iVar) {

  static unsigned PMasks[4][3] = {{0x99999999, 0x22222222, 0x44444444},
                                  {0xC3C3C3C3, 0x0C0C0C0C, 0x30303030},
                                  {0xF00FF00F, 0x00F000F0, 0x0F000F00},
                                  {0xFF0000FF, 0x0000FF00, 0x00FF0000}};

  int nWords = wordNum(nVars);
  int i, k, step, shift;

  assert(iVar < nVars - 1);

  if (iVar < 4) {
    shift = (1 << iVar);
    for (i = 0; i < nWords; i++) {
      result[i] = (input[i] & PMasks[iVar][0]) |
                  ((input[i] & PMasks[iVar][1]) << shift) |
                  ((input[i] & PMasks[iVar][2]) >> shift);
    }
  } else {
    if (iVar > 4) {
      step = (1 << (iVar - 5));
      for (k = 0; k < nWords; k += 4 * step) {

        for (i = 0; i < step; i++) {
          result[i] = input[i];
        }

        for (i = 0; i < step; i++) {
          result[step + i] = input[2 * step + i];
        }

        for (i = 0; i < step; i++) {
          result[2 * step + i] = input[step + i];
        }

        for (i = 0; i < step; i++) {
          result[3 * step + i] = input[3 * step + i];
        }

        input += 4 * step;
        result += 4 * step;
      }
    } else { // if ( iVar == 4 )
      for (i = 0; i < nWords; i += 2) {
        result[i] =
            (input[i] & 0x0000FFFF) | ((input[i + 1] & 0x0000FFFF) << 16);
        result[i + 1] =
            (input[i + 1] & 0xFFFF0000) | ((input[i] & 0xFFFF0000) >> 16);
      }
    }
  }
}

inline std::string toBin(word* function, int nWords) {

  if (function != nullptr) {

    std::stringstream result;

    result << "";

    for (int i = nWords - 1; i >= 0; i--) {
      for (int j = 31; j >= 0; j--) {
        if ((function[i] >> j) & 1) {
          result << ("1");
        } else {
          result << ("0");
        }
      }
    }

    return result.str();
  } else {
    return "nullptr";
  }
}
inline std::string toHex(word* function, int nWords) {

  std::stringstream result;

  result << "0x";

  for (int i = nWords - 1; i >= 0; i--) {
    result << std::setw(16) << std::setfill('0') << std::hex << function[i];
  }

  return result.str();
}

inline std::string toCubeString( word* function, int nWords, int nVars) {

	std::stringstream cubes;
	word mask, cube;
	int nRows;

	if ( nWords == 1 ) {
		nRows = 2 << (nVars-1);
		mask = 1;
		for (int j = 0; j < nRows; j++) {
			if ( function[0] & mask ) {
				cube = j;
				for (int k = 0; k < nVars; k++) {
					if ((cube >> k) & 1) {
						cubes << ("1");
					}
					else {
						cubes << ("0");
					}
				}
				cubes << " 1" << std::endl;
			}
			mask = mask << 1;
		}
	}
	else {
		for (int i = 0; i < nWords; i++) {
			mask = 1;
			for (int j = 0; j < 32; j++) {
				if ( function[i] & mask ) {
					cube = (i*32)+j;
					for (int k = 0; k < nVars; k++) {
						if ((cube >> k) & 1) {
							cubes << ("1");
						} 
						else {
							cubes << ("0");
						}
					}
					cubes << " 1" << std::endl;
				}
				mask = mask << 1;
			}
		}
	}
	return cubes.str();
}

} /* namespace Functional32 */

#endif /* FUNCTIONAL32_H_ */
