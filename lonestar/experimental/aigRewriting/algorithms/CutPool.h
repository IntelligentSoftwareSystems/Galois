/*

 @Vinicius Possani
 Parallel Rewriting January 5, 2018.
 ABC-based implementation on Galois.

*/

#ifndef CUTPOOL_H_
#define CUTPOOL_H_

#include <vector>

namespace algorithm {

// The size of the leaves is defined acording the parameter Kk, during the
// memory allocation in the CutPool.cpp
typedef struct cut_ {
  unsigned int sig;
  short int nLeaves;
  struct cut_* nextCut;
  int leaves[0];
} Cut;

class CutPool {

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
  CutPool(long int initialSize, int k, bool compTruth);

  ~CutPool();

  Cut* getMemory();

  void giveBackMemory(Cut* cut);

  int getNumBlocks();

  int getBlockSize();
};

} /* namespace algorithm */

#endif /* CUTPOOL_H_ */
