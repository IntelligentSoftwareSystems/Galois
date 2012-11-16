#ifndef GALOIS_FOREACH_H
#define GALOIS_FOREACH_H

#include "Galois/Runtime/TaskRuntime.h"

namespace Galois {

   template<typename IterTy, typename FunctionTy>
   void for_each_task(IterTy b, IterTy e, FunctionTy f) {
      GaloisRuntime::for_each_task_impl<IterTy,FunctionTy>(b, e, f);
      GaloisRuntime::set_distributed_foreach(false);
   }

   void for_each_begin() {
      GaloisRuntime::set_distributed_foreach(true);
      GaloisRuntime::for_each_begin_impl();
   }
}
#endif
