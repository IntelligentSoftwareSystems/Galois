#ifndef GALOIS_FOREACH_H
#define GALOIS_FOREACH_H

#include "Galois/Runtime/TaskRuntime.h"

namespace Galois {

   template<typename IterTy, typename FunctionTy>
   void for_each_task(IterTy b, IterTy e, FunctionTy f) {
      Galois::Runtime::for_each_task_impl<IterTy,FunctionTy>(b, e, f);
      Galois::Runtime::set_distributed_foreach(false);
   }

   void for_each_begin() {
      Galois::Runtime::set_distributed_foreach(true);
      Galois::Runtime::for_each_begin_impl();
   }
}
#endif
