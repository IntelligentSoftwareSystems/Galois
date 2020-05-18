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

/*

 @Vinicius Possani
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#ifndef PRIORITYCUTPOOL_H_
#define PRIORITYCUTPOOL_H_

#include <vector>

namespace algorithm {

// The size of the leaves is defined acording the parameter K, during the
// memory allocation in the CutPool.cpp
typedef struct priCut_ {
  float area;  // area (or area-flow) of the cut
  float edge;  // the edge flow
  float power; // the power flow
  float delay; // delay of the cut

  unsigned int sig;
  short int nLeaves;
  struct priCut_* nextCut;
  int leaves[0];
} PriCut;

class PriCutPool {

private:
  long int blockSize;
  int k;
  int entrySize;
  long int entriesUsed;
  long int entriesAlloc;
  char* entriesFree;
  std::vector<char*> blocks;

  void alloc();

public:
  PriCutPool(long int initialSize, int k, bool compTruth);

  ~PriCutPool();

  PriCut* getMemory();

  void giveBackMemory(PriCut* cut);

  int getNumBlocks();

  int getBlockSize();

  // void copyCut(PriCut* dest, PriCut* source);
};

} /* namespace algorithm */

#endif /* PRIORITYCUTPOOL_H_ */
