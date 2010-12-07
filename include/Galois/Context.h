// Uservisible galois context -*- C++ -*-

#ifndef _GALOIS_CONTEXT_H
#define _GALOIS_CONTEXT_H

#include "Galois/Executable.h"
#include "Galois/Runtime/mm/mem.h"

#include <boost/utility.hpp>

namespace GaloisRuntime {
class SimpleRuntimeContext;
}

namespace Galois {

template<typename T>
class Context : boost::noncopyable {
  //  typedef GaloisRuntime::MM::ZeroOut<GaloisRuntime::MM::SimpleBumpPtr<GaloisRuntime::MM::FreeListHeap<GaloisRuntime::MM::SystemBaseAlloc> > > ItAllocBaseTy;
  typedef GaloisRuntime::MM::SimpleBumpPtr<GaloisRuntime::MM::FreeListHeap<GaloisRuntime::MM::SystemBaseAlloc> > ItAllocBaseTy;

  ItAllocBaseTy IterationAllocatorBase;
protected:
  void resetAlloc() {
    IterationAllocatorBase.clear();
  }

public:
  Context()
    :IterationAllocatorBase(), 
     PerIterationAllocator(&IterationAllocatorBase)
  {}
  virtual ~Context() {
    IterationAllocatorBase.clear();
  }

  virtual GaloisRuntime::SimpleRuntimeContext* getRuntimeContext() = 0;

  virtual void push(T) = 0;
  virtual void finish() = 0;
  virtual void suspendWith(Executable*) = 0;
  
  typedef GaloisRuntime::MM::ExternRefGaloisAllocator<char, ItAllocBaseTy> ItAllocTy;
  ItAllocTy PerIterationAllocator;

};


}

#endif
