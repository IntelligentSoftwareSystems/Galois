#include "galois/runtime/Executor_Deterministic.h"

thread_local galois::runtime::SizedHeapFactory::SizedHeap* 
    galois::runtime::internal::dagListHeap;
