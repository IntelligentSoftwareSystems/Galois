#ifndef GALOIS_FOREACH_H
#define GALOIS_FOREACH_H

#include "galois/runtime/TaskRuntime.h"

namespace galois {

template <typename IterTy, typename FunctionTy>
void for_each_task(IterTy b, IterTy e, FunctionTy f) {
  galois::runtime::for_each_task_impl<IterTy, FunctionTy>(b, e, f);
  galois::runtime::set_distributed_foreach(false);
}

void for_each_begin() {
  galois::runtime::set_distributed_foreach(true);
  galois::runtime::for_each_begin_impl();
}
} // namespace galois
#endif
