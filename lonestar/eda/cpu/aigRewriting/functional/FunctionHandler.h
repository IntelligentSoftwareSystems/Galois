/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef FUNCTIONAL_H_
#define FUNCTIONAL_H_

#include "../xxHash/xxhash.h"
#include <cmath>
#include <cstring>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <bitset>
#include <cassert>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "BitVectorPool.h"

namespace Functional {

typedef unsigned long word;

enum Order { SMALLER, LARGER, NOTCOMPARABLE, EQUAL };

inline void createLiterals(
    std::vector<std::string>& varSet,
    std::unordered_map<std::string, std::pair<word*, unsigned int>>& literals,
    BitVectorPool& functionPool);
inline bool less(word* lhs, word* rhs, int nWords);
inline bool equals(word* lhs, word* rhs, int nWords);
inline bool diff(word* lhs, word* rhs, int nWords);
inline void copy(word* result, word* original, int nWords);
inline void NOT(word* result, word* original, int nWords);
inline void AND(word* result, word* lhs, word* rhs, int nWords);
inline void OR(word* result, word* lhs, word* rhs, int nWords);
inline void XOR(word* result, word* lhs, word* rhs, int nWords);
inline void MUX(word* result, word* zero, word* one, word* sel, int nWords);
inline void cofactor0(word* result, word* original, int nWords, int iVar);
inline void cofactor1(word* result, word* original, int nWords, int iVar);
inline int getSupport(word* function, int nVars);
inline int getPolarizedSupport(word* function, int nVars);
inline bool hasVar(word* functin, int nVars, int iVar);
inline bool hasVarTruth6(word* function, int iVar);
inline bool posVar6(word t, int iVar);
inline bool negVar6(word t, int iVar);
inline bool posVar(word* function, int nVars, int iVar);
inline bool negVar(word* function, int nVars, int iVar);
inline bool isUnate(word* function, int nVars);
inline bool isPosUnate(word* function, int nVars);
inline bool isConstZero(word* function, int nVars);
inline bool isConstOne(word* function, int nVars);
inline Order order(word* sub, word* target, int nWords);
inline int getHammingDist(word* f1, word* f2, int nWords);
inline int oneCounter(unsigned long int word);
inline int wordNum(int nVars);
inline bool isOdd(word* function);
inline std::string toBin(word* function, int nWords);
inline std::string toHex(word* function, int nWords);
inline std::string supportToBin(unsigned int support);

inline constexpr word truths6[6] = {0xAAAAAAAAAAAAAAAA, 0xCCCCCCCCCCCCCCCC,
                                    0xF0F0F0F0F0F0F0F0, 0xFF00FF00FF00FF00,
                                    0xFFFF0000FFFF0000, 0xFFFFFFFF00000000};

inline constexpr word truths6Neg[6] = {0x5555555555555555, 0x3333333333333333,
                                       0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF,
                                       0x0000FFFF0000FFFF, 0x00000000FFFFFFFF};

class FunctionHasher {
  int nWords;
  size_t TRUTH_WORDS_BYTE_COUNT;

public:
  FunctionHasher(int nWords) : nWords(nWords) {
    TRUTH_WORDS_BYTE_COUNT = sizeof(Functional::word) * nWords;
  }

  size_t operator()(const Functional::word* function) const {

    if (nWords == 1) {
      return function[0];
    } else {
      return XXH64(function, TRUTH_WORDS_BYTE_COUNT, 0);
    }
  }
};

class FunctionComparator {
  int nWords;

public:
  FunctionComparator(int nWords) : nWords(nWords) {}

  bool operator()(Functional::word* f1, Functional::word* f2) const {
    return Functional::equals(f1, f2, nWords);
  }
};

typedef struct functionData {
  unsigned int support;
  unsigned int occurrences;
} FunctionData;

using FunctionSet =
    std::unordered_set<Functional::word*, FunctionHasher, FunctionComparator>;
using FunctionDataMap =
    std::unordered_map<word*, FunctionData, FunctionHasher, FunctionComparator>;

inline void computeAllCubeCofactors(BitVectorPool& functionPool,
                                    FunctionSet& cubeCofactors, word* function,
                                    int nVars);
inline void computeAllCubeCofactorsRec(BitVectorPool& functionPool,
                                       FunctionSet& cubeCofactors,
                                       word* function, int nVars, int nWords,
                                       int iVar, bool isPrevVarCofactored);
inline void
computeAllCubeCofactorsWithSupport(BitVectorPool& functionPool,
                                   FunctionDataMap& cubeCofactorData,
                                   word* function, int nVars);
inline void computeAllCubeCofactorsWithSupportRec(
    BitVectorPool& functionPool, FunctionDataMap& cubeCofactorData,
    word* function, int nVars, int nWords, int iVar);
inline void registerFunction(FunctionDataMap& cubeCofactorData, word* function,
                             int nVars, unsigned int occurencesInc);

inline void createLiterals(
    std::vector<std::string>& varSet,
    std::unordered_map<std::string, std::pair<word*, unsigned int>>& literals,
    BitVectorPool& functionPool) {

  int nVars  = varSet.size();
  int nWords = wordNum(nVars);

  for (int iVar = 0; iVar < nVars; iVar++) {

    word* currentFunction = functionPool.getMemory();

    if (iVar < 6) {
      for (int k = 0; k < nWords; k++) {
        currentFunction[k] = truths6[iVar];
      }
    } else {
      for (int k = 0; k < nWords; k++) {
        currentFunction[k] = (k & (1 << (iVar - 6))) ? ~(word)0 : 0;
      }
    }

    unsigned int support = getPolarizedSupport(currentFunction, nVars);
    literals.insert(
        std::make_pair(varSet[iVar], std::make_pair(currentFunction, support)));

    // Create negative literals
    std::string negLit = "!" + varSet[iVar];
    word* negFunction  = functionPool.getMemory();
    NOT(negFunction, currentFunction, nWords);
    unsigned int negSupport = (support >> 1);
    literals.insert(
        std::make_pair(negLit, std::make_pair(negFunction, negSupport)));
  }

  // Create constant zero and constant one
  word* constZero = functionPool.getMemory();
  word* constOne  = functionPool.getMemory();
  for (int k = 0; k < nWords; k++) {
    constZero[k] = 0UL;
    constOne[k]  = ~0UL;
  }
  literals.insert(std::make_pair("0", std::make_pair(constZero, 0)));
  literals.insert(std::make_pair("1", std::make_pair(constOne, 0)));

  //	std::cout << std::endl << "############################## Literals
  //##############################" << std::endl; 	for ( auto lit : literals )
  //{ 		std::cout << lit.first << " = " << toHex( lit.second.first, nWords )
  //<< " | " << supportToBin( lit.second.second ) << std::endl;
  //	}
  //	std::cout << std::endl;
}

inline bool less(word* lhs, word* rhs, int nWords) {

  if ((lhs == nullptr) || (rhs == nullptr)) {
    return false;
  }

  for (int i = nWords - 1; i >= 0; --i) {
    if (lhs[i] < rhs[i]) {
      return true;
    }
    if (lhs[i] > rhs[i]) {
      return false;
    }
  }

  return false;
}

inline bool equals(word* lhs, word* rhs, int nWords) {

  if ((lhs == nullptr) || (rhs == nullptr)) {
    return false;
  }

  for (int i = 0; i < nWords; i++) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }

  return true;
}

inline bool diff(word* lhs, word* rhs, int nWords) {

  if ((lhs == nullptr) || (rhs == nullptr)) {
    return false;
  }

  for (int i = 0; i < nWords; i++) {
    if (lhs[i] != rhs[i]) {
      return true;
    }
  }

  return false;
}

inline void copy(word* result, word* original, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = original[i];
  }
}

inline void NOT(word* result, word* original, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = ~(original[i]);
  }
}

inline void AND(word* result, word* lhs, word* rhs, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = lhs[i] & rhs[i];
  }
}

inline void OR(word* result, word* lhs, word* rhs, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = lhs[i] | rhs[i];
  }
}

inline void XOR(word* result, word* lhs, word* rhs, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = lhs[i] ^ rhs[i];
  }
}

inline void MUX(word* result, word* zero, word* one, word* sel, int nWords) {

  for (int i = 0; i < nWords; i++) {
    result[i] = (zero[i] & ~sel[i]) | (one[i] & sel[i]);
  }
}

inline void cofactor0(word* result, word* original, int nWords, int iVar) {

  if (nWords == 1) {
    result[0] = ((original[0] & truths6Neg[iVar]) << (1 << iVar)) |
                (original[0] & truths6Neg[iVar]);
  } else {
    if (iVar <= 5) {
      int w, shift = (1 << iVar);
      for (w = 0; w < nWords; w++) {
        result[w] = ((original[w] & truths6Neg[iVar]) << shift) |
                    (original[w] & truths6Neg[iVar]);
      }
    } else { // if ( iVar > 5 )
      word* pOriginal = original;
      word* pResult   = result;

      word* pLimit = pOriginal + nWords;
      int i, iStep = wordNum(iVar);
      for (; pOriginal < pLimit; pOriginal += 2 * iStep, pResult += 2 * iStep) {
        for (i = 0; i < iStep; i++) {
          pResult[i]         = pOriginal[i];
          pResult[i + iStep] = pOriginal[i];
        }
      }
    }
  }
}

inline void cofactor1(word* result, word* original, int nWords, int iVar) {

  if (nWords == 1) {
    result[0] = (original[0] & truths6[iVar]) |
                ((original[0] & truths6[iVar]) >> (1 << iVar));
  } else {
    if (iVar <= 5) {
      int w, shift = (1 << iVar);
      for (w = 0; w < nWords; w++) {
        result[w] = (original[w] & truths6[iVar]) |
                    ((original[w] & truths6[iVar]) >> shift);
      }
    } else { // if ( iVar > 5 )
      word* pOriginal = original;
      word* pResult   = result;

      word* pLimit = pOriginal + nWords;
      int i, iStep = wordNum(iVar);
      for (; pOriginal < pLimit; pOriginal += 2 * iStep, pResult += 2 * iStep) {
        for (i = 0; i < iStep; i++) {
          pResult[i]         = pOriginal[i + iStep];
          pResult[i + iStep] = pOriginal[i + iStep];
        }
      }
    }
  }
}

inline void computeAllCubeCofactors(BitVectorPool& functionPool,
                                    FunctionSet& cubeCofactors, word* function,
                                    int nVars) {
  bool isPrevVarCofactored = true;
  int nWords               = wordNum(nVars);
  int iVar                 = nVars - 1;
  computeAllCubeCofactorsRec(functionPool, cubeCofactors, function, nVars,
                             nWords, iVar, isPrevVarCofactored);
}

inline void computeAllCubeCofactorsRec(BitVectorPool& functionPool,
                                       FunctionSet& cubeCofactors,
                                       word* function, int nVars, int nWords,
                                       int iVar, bool isPrevVarCofactored) {

  // ignoring constants
  if (isConstZero(function, nVars) || isConstOne(function, nVars)) {
    return;
  }

  if (isPrevVarCofactored) {
    // inserting the current function to the unique set
    auto status = cubeCofactors.insert(function);
    // When the function was already visited
    if (status.second == false) {
      functionPool.giveBackMemory();
      return;
    }
  }

  // When the terminal case is found
  if (iVar < 0) {
    return;
  }

  // Calling recursing with iVar as dont care
  computeAllCubeCofactorsRec(functionPool, cubeCofactors, function, nVars,
                             nWords, iVar - 1, false);

  // When iVar is dont care
  if (hasVar(function, nVars, iVar) == false) {
    return;
  }

  // Calling recursing with iVar = 0
  word* negCof = functionPool.getMemory();
  cofactor0(negCof, function, nWords, iVar);
  computeAllCubeCofactorsRec(functionPool, cubeCofactors, negCof, nVars, nWords,
                             iVar - 1, true);

  // Calling recursing with iVar = 1
  word* posCof = functionPool.getMemory();
  cofactor1(posCof, function, nWords, iVar);
  computeAllCubeCofactorsRec(functionPool, cubeCofactors, posCof, nVars, nWords,
                             iVar - 1, true);
}

inline void
computeAllCubeCofactorsWithSupport(BitVectorPool& functionPool,
                                   FunctionDataMap& cubeCofactorData,
                                   word* function, int nVars) {
  int nWords = wordNum(nVars);
  int iVar   = nVars - 1;
  computeAllCubeCofactorsWithSupportRec(functionPool, cubeCofactorData,
                                        function, nVars, nWords, iVar);
}

inline void computeAllCubeCofactorsWithSupportRec(
    BitVectorPool& functionPool, FunctionDataMap& cubeCofactorData,
    word* function, int nVars, int nWords, int iVar) {

  // When the constants are found
  if (isConstZero(function, nVars) || isConstOne(function, nVars)) {
    unsigned int occurrencesInc = (unsigned int)pow(3, iVar + 1);
    registerFunction(cubeCofactorData, function, nVars, occurrencesInc);
    return;
  }

  // When the terminal case is found
  if (iVar < 0) {
    unsigned int occurrencesInc = 1;
    registerFunction(cubeCofactorData, function, nVars, occurrencesInc);
    return;
  }

  // Calling recursing with iVar as dont care
  computeAllCubeCofactorsWithSupportRec(functionPool, cubeCofactorData,
                                        function, nVars, nWords, iVar - 1);

  // Calling recursing with iVar = 0
  word* negCof = functionPool.getMemory();
  cofactor0(negCof, function, nWords, iVar);
  computeAllCubeCofactorsWithSupportRec(functionPool, cubeCofactorData, negCof,
                                        nVars, nWords, iVar - 1);

  // Calling recursing with iVar = 1
  word* posCof = functionPool.getMemory();
  cofactor1(posCof, function, nWords, iVar);
  computeAllCubeCofactorsWithSupportRec(functionPool, cubeCofactorData, posCof,
                                        nVars, nWords, iVar - 1);
}

inline void registerFunction(FunctionDataMap& cubeCofactorData, word* function,
                             int nVars, unsigned int occurencesInc) {

  auto it = cubeCofactorData.find(function);
  if (it == cubeCofactorData.end()) {
    FunctionData functionData;
    // functionData.support = getSupport( function, nVars );
    functionData.support     = getPolarizedSupport(function, nVars);
    functionData.occurrences = occurencesInc;
    cubeCofactorData.insert(std::make_pair(function, functionData));
  } else {
    it->second.occurrences = it->second.occurrences + occurencesInc;
  }
}

inline bool hasVar(word* function, int nVars, int iVar) {

  word* t    = function;
  int nWords = wordNum(nVars);

  if (nWords == 1) {
    return hasVarTruth6(function, iVar);
  }
  if (iVar < 6) {
    int i, Shift = (1 << iVar);
    for (i = 0; i < nWords; i++) {
      if (((t[i] >> Shift) & truths6Neg[iVar]) != (t[i] & truths6Neg[iVar])) {
        return true;
      }
    }
    return false;
  } else {
    int i, Step = (1 << (iVar - 6));
    word* tLimit = t + nWords;
    for (; t < tLimit; t += 2 * Step) {
      for (i = 0; i < Step; i++) {
        if (t[i] != t[Step + i]) {
          return true;
        }
      }
    }
    return false;
  }
}

inline bool hasVarTruth6(word* function, int iVar) {
  word t = function[0];
  return ((t >> (1 << iVar)) & truths6Neg[iVar]) != (t & truths6Neg[iVar]);
}

inline bool posVar6(word t, int iVar) {
  return ((t >> (1 << iVar)) & t & truths6Neg[iVar]) == (t & truths6Neg[iVar]);
}

inline bool negVar6(word t, int iVar) {
  return ((t << (1 << iVar)) & t & truths6[iVar]) == (t & truths6[iVar]);
}

inline bool posVar(word* function, int nVars, int iVar) {

  assert(iVar < nVars);

  word* t = function;

  if (nVars <= 6) {
    return posVar6(t[0], iVar);
  }
  if (iVar < 6) {
    int i, shift = (1 << iVar);
    int nWords = wordNum(nVars);
    for (i = 0; i < nWords; i++) {
      if (((t[i] >> shift) & t[i] & truths6Neg[iVar]) !=
          (t[i] & truths6Neg[iVar])) {
        return false;
      }
    }
    return true;
  } else {
    int i, step = (1 << (iVar - 6));
    word* tLimit = t + wordNum(nVars);
    for (; t < tLimit; t += 2 * step) {
      for (i = 0; i < step; i++) {
        if (t[i] != (t[i] & t[step + i])) {
          return false;
        }
      }
    }
    return true;
  }
}

inline bool negVar(word* function, int nVars, int iVar) {

  assert(iVar < nVars);

  word* t = function;

  if (nVars <= 6) {
    return negVar6(t[0], iVar);
  }
  if (iVar < 6) {
    int i, shift = (1 << iVar);
    int nWords = wordNum(nVars);
    for (i = 0; i < nWords; i++) {
      if (((t[i] << shift) & t[i] & truths6[iVar]) != (t[i] & truths6[iVar])) {
        return false;
      }
    }
    return true;
  } else {
    int i, step = (1 << (iVar - 6));
    word* tLimit = t + wordNum(nVars);
    for (; t < tLimit; t += 2 * step) {
      for (i = 0; i < step; i++) {
        if ((t[i] & t[step + i]) != t[step + i]) {
          return false;
        }
      }
    }
    return true;
  }
}

inline bool isUnate(word* function, int nVars) {

  for (int i = 0; i < nVars; i++) {
    if (!negVar(function, nVars, i) && !posVar(function, nVars, i)) {
      return false;
    }
  }
  return true;
}

inline bool isPosUnate(word* function, int nVars) {

  for (int i = 0; i < nVars; i++) {
    if (!posVar(function, nVars, i)) {
      return false;
    }
  }
  return true;
}

inline int getSupport(word* function, int nVars) {

  int v, Supp = 0;
  for (v = 0; v < nVars; v++) {
    if (hasVar(function, nVars, v)) {
      Supp |= (1 << v);
    }
  }
  return Supp;
}

inline int getPolarizedSupport(word* function, int nVars) {

  int v, Supp = 0;
  for (v = 0; v < nVars; v++) {
    if (!posVar(function, nVars, v)) {
      Supp |= (1 << (v * 2));
    }
    if (!negVar(function, nVars, v)) {
      Supp |= (1 << ((v * 2) + 1));
    }
  }
  return Supp;
}

inline bool isConstZero(word* function, int nVars) {

  word* pFunction = function;
  word* pLimit    = pFunction + wordNum(nVars);

  while (pFunction != pLimit) {
    if (*pFunction++ != 0ULL) {
      return false;
    }
  }

  return true;
}

inline bool isConstOne(word* function, int nVars) {

  word* pFunction = function;
  word* pLimit    = pFunction + wordNum(nVars);
  const word ONE  = ~0ULL;

  while (pFunction != pLimit) {
    if (*pFunction++ != ONE) {
      return false;
    }
  }

  return true;
}

inline Order order(word* sub, word* target, int nWords) {

  if (equals(sub, target, nWords)) {
    return Order::EQUAL;
  }

  bool smaller = true;
  bool larger  = true;
  unsigned long int partialResult;

  for (int i = 0; i < nWords; i++) {

    partialResult = sub[i] & target[i];

    if (partialResult != sub[i]) {
      smaller = false;
    }
    if (partialResult != target[i]) {
      larger = false;
    }

    if (!smaller && !larger) {
      return Order::NOTCOMPARABLE;
    }
  }

  if (smaller)
    return Order::SMALLER;

  if (larger)
    return Order::LARGER;

  assert(false); // Should never happen
  return Order::NOTCOMPARABLE;
}

inline int getHammingDist(word* f1, word* f2, int nWords) {

  unsigned long int currentWord;
  int count = 0;

  for (int i = 0; i < nWords; i++) {
    currentWord = f1[i] ^ f2[i];
    count += oneCounter(currentWord);
  }

  return count;
}

// This is better when most bits in word are 0. It uses 3 arithmetic operations
// and one comparison/branch per "1" bit in word.
inline int oneCounter(unsigned long int word) {

  int count;
  for (count = 0; word; count++) {
    word &= word - 1;
  }
  return count;
}

inline int wordNum(int nVars) { return nVars <= 6 ? 1 : 1 << (nVars - 6); }

inline bool isOdd(word* function) { return (function[0] & 1) != 0; }

inline std::string toBin(word* function, int nWords) {

  if (function != nullptr) {

    std::stringstream result;

    result << "";

    for (int i = nWords - 1; i >= 0; i--) {
      for (int j = 63; j >= 0; j--) {
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

inline std::string supportToBin(unsigned int support) {
  word ptr[1];
  ptr[0] = support;
  return Functional::toBin(ptr, 1);
}

} /* namespace Functional */

#endif /* FUNCTIONAL_H_ */
