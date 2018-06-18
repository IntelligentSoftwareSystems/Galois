/*
 * FunctionPool.cpp
 *
 *  Created on: 27/03/2017
 *      Author: viniciuspossani
 */

#include <iostream>
#include "BitVectorPool.h"

namespace Functional {

BitVectorPool::BitVectorPool(int nElements, int nWords) {
  this->blockSize    = nElements;
  this->nWords       = nWords;
  this->index        = 0;
  this->currentBlock = nullptr;
  alloc();
}

BitVectorPool::~BitVectorPool() {
  //	std::cout << "Deleting Blocks..." << std::endl;
  for (auto ptr : this->blocks) {
    free(ptr[0]);
    free(ptr);
  }
}

void BitVectorPool::alloc() {

  word* tmp = (word*)malloc(sizeof(word) * (this->blockSize * this->nWords));
  if (tmp == nullptr) {
    std::cout << "Error: memory could not be allocated by BitVectorPool!"
              << std::endl;
    exit(1);
  }

  this->currentBlock = (word**)malloc(sizeof(word*) * this->blockSize);
  if (this->currentBlock == nullptr) {
    std::cout << "Error: memory could not be allocated by BitVectorPool!"
              << std::endl;
    exit(1);
  }

  int i, j = 0;
  for (i = 0; i < this->blockSize; i++) {
    this->currentBlock[i] = &tmp[j];
    j += this->nWords;
  }

  this->blocks.push_back(this->currentBlock);
}

word* BitVectorPool::getMemory() {

  if (index >= blockSize) {
    alloc();
    this->index = 0;
  }

  word* ptr = this->currentBlock[this->index];
  this->index++;
  return ptr;
}

word* BitVectorPool::getCleanMemory() {

  if (index >= blockSize) {
    alloc();
    this->index = 0;
  }

  for (int i = 0; i < this->nWords; i++) {
    this->currentBlock[this->index][i] = 0;
  }

  word* ptr = this->currentBlock[this->index];
  this->index++;
  return ptr;
}

void BitVectorPool::giveBackMemory() { this->index--; }

int BitVectorPool::getNumBlocks() { return this->blocks.size(); }

} /* namespace Functional */
