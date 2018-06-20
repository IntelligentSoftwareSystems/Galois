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
 * FunctionPool.h
 *
 *  Created on: 27/03/2017
 *      Author: viniciuspossani
 */

#ifndef SRC_MAIN_FUNCTIONPOOL_H_
#define SRC_MAIN_FUNCTIONPOOL_H_

#include <vector>

namespace Functional {

typedef unsigned long word;

class BitVectorPool {

  int blockSize;
  int nWords;
  int index;
  word** currentBlock;
  std::vector<word**> blocks;

  void alloc();

public:
  BitVectorPool(int nElements, int nWords);

  virtual ~BitVectorPool();

  word* getMemory();

  word* getCleanMemory();

  void giveBackMemory();

  int getNumBlocks();
};

} /* namespace Functional */

#endif /* SRC_MAIN_FUNCTIONPOOL_H_ */
