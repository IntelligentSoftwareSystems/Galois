/*

 @Vinicius Possani
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#include "CutPool.h"
#include "../functional/FunctionHandler32.h"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cassert>

namespace algorithm {

CutPool::CutPool(long int initialSize, int k, bool compTruth) {
  this->blockSize = initialSize;
  this->k         = k;
  if (compTruth) {
    this->entrySize = sizeof(Cut) + (k * sizeof(int)) +
                      (Functional32::wordNum(k) * sizeof(unsigned int));
  } else {
    this->entrySize = sizeof(Cut) + (k * sizeof(int));
  }
  this->entriesUsed  = 0;
  this->entriesAlloc = 0;
  this->entriesFree  = nullptr;
}

CutPool::~CutPool() {
  for (char* ptr : this->blocks) {
    free(ptr);
  }
}

inline void CutPool::alloc() {

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

Cut* CutPool::getMemory() {

  if (this->entriesUsed == this->entriesAlloc) {
    assert(this->entriesFree == nullptr);
    alloc();
  }

  this->entriesUsed++;
  char* pTemp       = this->entriesFree;
  this->entriesFree = *((char**)pTemp);

  Cut* cut = (Cut*)pTemp;
  memset(cut, 0, this->entrySize);
  cut->nextCut = nullptr;

  return cut;
}

void CutPool::giveBackMemory(Cut* cut) {

  this->entriesUsed--;
  char* pTemp       = (char*)cut;
  *((char**)pTemp)  = this->entriesFree;
  this->entriesFree = pTemp;
}

int CutPool::getNumBlocks() { return this->blocks.size(); }

int CutPool::getBlockSize() { return this->blockSize; }

} /* namespace algorithm */
