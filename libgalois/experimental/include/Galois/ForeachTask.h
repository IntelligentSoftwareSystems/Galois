#ifndef GALOIS_FOREACH_H
#define GALOIS_FOREACH_H

#include "Galois/Runtime/TaskRuntime.h"

namespace galois {

   template<typename IterTy, typename FunctionTy>
   void for_each_task(IterTy b, IterTy e, FunctionTy f) {
      galois::Runtime::for_each_task_impl<IterTy,FunctionTy>(b, e, f);
      galois::Runtime::set_distributed_foreach(false);
   }

   void for_each_begin() {
      galois::Runtime::set_distributed_foreach(true);
      galois::Runtime::for_each_begin_impl();
   }
}
#endif
