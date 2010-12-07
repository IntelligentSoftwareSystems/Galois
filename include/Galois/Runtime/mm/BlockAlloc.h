// -*- C++ -*-

#include "Support/ThreadSafe/simple_lock.h"
#include "mem.h"

#include <vector>

namespace GaloisRuntime {
namespace MM {

template<typename Ty, typename BaseAlloc = SystemBaseAlloc>
class TSBlockAlloc : public BaseAlloc {

  struct TyEq {
    size_t data[(sizeof(Ty) + sizeof(size_t) - 1) / sizeof(size_t)];
  };

  struct cpuBlock {
    TyEq* block;
    int headIndex;
    cpuBlock() :block(0), headIndex(-1) {}
  };

  enum { NumInBlock = BaseAlloc::AllocSize / sizeof(TyEq) };

  std::vector<void*> blocks;
  threadsafe::simpleLock<int, true> blocksLock;

  GaloisRuntime::CPUSpaced<cpuBlock> AllocList;

  static void merge(cpuBlock& lhs, cpuBlock& rhs) {
  }

  void refill(cpuBlock& B) {
    void* P = BaseAlloc::allocate(BaseAlloc::AllocSize);
    blocksLock.lock();
    blocks.push_back(P);
    blocksLock.unlock();
    B.block = (TyEq*)P;
    B.headIndex = 0;
  }
 

public:
  TSBlockAlloc() 
    :AllocList(merge)
  {}

  ~TSBlockAlloc() {
    for (unsigned int i = 0; i < blocks.size(); ++i)
      BaseAlloc::deallocate(blocks[i]);
  }

  Ty* allocate(unsigned int) {
    cpuBlock& B = AllocList.get();
    if (B.block == 0 || B.headIndex == NumInBlock) {
      refill(B);
    }
    TyEq* ptr = &B.block[B.headIndex];
    ++B.headIndex;
    return (Ty*)ptr;
  }

  void deallocate(Ty*) {
  }
};

}
}
