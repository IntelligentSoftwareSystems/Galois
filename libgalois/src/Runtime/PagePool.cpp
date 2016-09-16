/** Page Pool  Implementation -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/PagePool.h"
#include "Galois/Runtime/PageAlloc.h"
#include "Galois/Runtime/SimpleLock.h"
#include "Galois/Runtime/PtrLock.h"
#include "Galois/Runtime/ThreadPool.h"
#include "Galois/Runtime/FlatMap.h"
#include "Galois/Runtime/PerThreadStorage.h"

#include <mutex>
#include <deque>

namespace {
struct FreeNode {
  FreeNode* next;
};
 
// Tracks pages allocated
class PAState {
  std::atomic<unsigned long> counts;
  Galois::Runtime::PtrLock<FreeNode> HeadPtr;

  static Galois::Runtime::SimpleLock lock;
  static Galois::Runtime::flat_map<void*, unsigned> owners;
  static void registerOwner(void* ptr, unsigned tid) {
    std::lock_guard<decltype(lock)> lg(lock);
    owners.insert(std::make_pair(ptr, tid));
  }
  
  void* allocFromOS() {
    void* ptr = Galois::Runtime::allocPages(1, true);
    assert(ptr);
    counts += 1;
    registerOwner(ptr, Galois::Runtime::ThreadPool::getTID());
    return ptr;
  }

public:
  unsigned long count() const {
    return counts;
  }

  void* pageAlloc() {
    if (HeadPtr.getValue()) {
      HeadPtr.lock();
      FreeNode* h = HeadPtr.getValue();
      if (h) {
        HeadPtr.unlock_and_set(h->next);
        return h;
      }
      HeadPtr.unlock();
    }
    return allocFromOS();
  }
  
  void pageFree(void* ptr) {
    HeadPtr.lock();
    FreeNode* nh = reinterpret_cast<FreeNode*>(ptr);
    nh->next = HeadPtr.getValue();
    HeadPtr.unlock_and_set(nh);
  }

  void pagePreAlloc() {
    pageFree(allocFromOS());
  }
  
  static unsigned getOwner(void* ptr) {
    std::lock_guard<decltype(lock)> lg(lock);
    auto ii = owners.find(ptr);
    if (ii == owners.end())
      return ~0;
    return ii->second;
  }
    
};

static Galois::Runtime::PerThreadStorage<PAState> PA;
} //end namespace ""

std::vector<unsigned long> Galois::Runtime::numPagePoolAllocTotal() {
  auto x = PA.size();
  std::vector<unsigned long> retval(x);
  for(unsigned y = 0; y < x; ++y)
    retval[y] = PA.get(y)->count();
  return retval;
}

void* Galois::Runtime::pagePoolAlloc() {
  return PA->pageAlloc();
}

void Galois::Runtime::pagePoolPreAlloc(unsigned num) {
  while (num--)
    PA->pagePreAlloc();
}

void Galois::Runtime::pagePoolFree(void* ptr) {
  PA->pageFree(ptr);
}

size_t Galois::Runtime::pagePoolSize() {
  return allocSize();
}

