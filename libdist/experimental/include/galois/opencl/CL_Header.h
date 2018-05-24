#ifdef __APPLE__
#include <opencl/opencl.h>
#else
extern "C" {
#include "CL/cl.h"
}
;
#endif

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/stat.h>
#include <assert.h>
#include "galois/opencl/CL_Errors.h"
#include "galois/opencl/CL_Util.h"
#include "galois/opencl/CL_Context.h"
#include "galois/opencl/CL_DeviceManager.h"
#include "galois/opencl/arraysArrays.h"
#include "galois/opencl/graphs/Graphs.h"
//////////////////////////////////////////
#ifndef GOPT_CL_HEADER_H_
#define GOPT_CL_HEADER_H_
namespace galois{
namespace opencl{
CLContext * getCLContext(){
   static CLContext ctx;
   return &ctx;
}
/*
 * Template wrapper for do_all_cl implementation.
 * */
template<typename ItTy, typename OpType, typename ... Args>
void do_all_cl(const ItTy & s, const ItTy & e, const OpType & f, const Args & ... args) {
   auto num_items = std::distance(s, e);
   auto * kernel = f.get_kernel(num_items); //(getCLContext()->get_default_device());
   kernel->set_work_size(num_items);
   (*kernel)();
}
}
}

#endif /* GOPT_CL_HEADER_H_ */
