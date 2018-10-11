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

 @Vinicius Possani
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#include "PriorityCutPool.h"
#include "../functional/FunctionHandler32.h"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cassert>

namespace algorithm {

PriCutPool::PriCutPool(long int initialSize, int k, bool compTruth) {
  this->blockSize = initialSize;
  this->k         = k;
  if (compTruth) {
    this->entrySize = sizeof(PriCut) + (k * sizeof(int)) +
                      (Functional32::wordNum(k) * sizeof(unsigned int));
  } else {
    this->entrySize = sizeof(PriCut) + (k * sizeof(int));
  }
  this->entriesUsed  = 0;
  this->entriesAlloc = 0;
  this->entriesFree  = nullptr;
}

PriCutPool::~PriCutPool() {
  for (char* ptr : this->blocks) {
    free(ptr);
  }
}

inline void PriCutPool::alloc() {

  this->entriesFree =
      (char*)malloc((long int)(this->entrySize * this->blockSize));

  if (this->entriesFree == nullptr) {
    std::cout << "Error: memory could not be allocated by CutPool!"
              << std::endl;
    exit(1);
  }

  char* pTemp = this->entriesFree;

  for (int i = 1; i < this->blockSize; i++) {
    *((char**)pTemp) = pTemp + this->entrySize;
    pTemp += this->entrySize;
  }

  *((char**)pTemp) = nullptr;

  this->entriesAlloc += this->blockSize;
  this->blocks.push_back(this->entriesFree);
}

PriCut* PriCutPool::getMemory() {

  if (this->entriesUsed == this->entriesAlloc) {
    assert(this->entriesFree == nullptr);
    alloc();
  }

  this->entriesUsed++;
  char* pTemp       = this->entriesFree;
  this->entriesFree = *((char**)pTemp);

  PriCut* cut = (PriCut*)pTemp;
  memset(cut, 0, this->entrySize);
  cut->nextCut = nullptr;

  return cut;
}

void PriCutPool::giveBackMemory(PriCut* cut) {

  this->entriesUsed--;
  char* pTemp       = (char*)cut;
  *((char**)pTemp)  = this->entriesFree;
  this->entriesFree = pTemp;
}

int PriCutPool::getNumBlocks() { return this->blocks.size(); }

int PriCutPool::getBlockSize() { return this->blockSize; }

} /* namespace algorithm */
