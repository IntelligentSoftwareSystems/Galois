// The user interface -*- C++ -*-

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

}
