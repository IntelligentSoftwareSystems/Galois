// Uservisible galois context -*- C++ -*-

#ifndef _GALOIS_PERITERMEM_H
#define _GALOIS_PERITERMEM_H

#include "Galois/Runtime/mm/mem.h"

#include <boost/utility.hpp>

namespace Galois {

class PerIterMem : boost::noncopyable {
  //  typedef GaloisRuntime::MM::ZeroOut<GaloisRuntime::MM::SimpleBumpPtr<GaloisRuntime::MM::FreeListHeap<GaloisRuntime::MM::SystemBaseAlloc> > > ItAllocBaseTy;
  typedef GaloisRuntime::MM::SimpleBumpPtr<GaloisRuntime::MM::FreeListHeap<GaloisRuntime::MM::SystemBaseAlloc> > ItAllocBaseTy;

  ItAllocBaseTy IterationAllocatorBase;
protected:
  void __resetAlloc() {
    IterationAllocatorBase.clear();
  }

public:
  PerIterMem()
    :IterationAllocatorBase(), 
     PerIterationAllocator(&IterationAllocatorBase)
  {}

  virtual ~PerIterMem() {
    IterationAllocatorBase.clear();
  }

  typedef GaloisRuntime::MM::ExternRefGaloisAllocator<char, ItAllocBaseTy> ItAllocTy;
  ItAllocTy PerIterationAllocator;
};


}

#endif
