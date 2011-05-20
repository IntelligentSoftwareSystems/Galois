// Uservisible galois context -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

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
