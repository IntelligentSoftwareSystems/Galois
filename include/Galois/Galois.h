// The user interface -*- C++ -*-
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

#include "Galois/Runtime/ParallelWork.h"

namespace Galois {

void setMaxThreads(unsigned int num);

template<typename Function, typename GWLTy>
void for_each(GWLTy& P, Function f) {
  GaloisRuntime::for_each_parallel(P, f);
}

template<typename IterTy, typename Function>
void for_each(IterTy b, IterTy e, Function f) {
  typedef GaloisRuntime::WorkList::ChunkedFIFO<typename IterTy::value_type, 256> GWLTy;
  GWLTy GWL;
  GWL.fill_initial(b, e);
  for_each(GWL, f);
}

template<typename Function>
void for_all(long start, long end, Function f) {
  GaloisRuntime::for_all_parallel(start, end, f);
}

template<typename Function>
void for_all(Function f) {
  int numThreads = GaloisRuntime::getSystemThreadPool().getActiveThreads();
  for_all(0, numThreads, f);
}

}
