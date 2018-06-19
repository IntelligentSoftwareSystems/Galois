/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#ifdef __APPLE__
#include <opencl/opencl.h>
#else
extern "C" {
#include "CL/cl.h"
};
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
namespace galois {
namespace opencl {
CLContext* getCLContext() {
  static CLContext ctx;
  return &ctx;
}
/*
 * Template wrapper for do_all_cl implementation.
 * */
template <typename ItTy, typename OpType, typename... Args>
void do_all_cl(const ItTy& s, const ItTy& e, const OpType& f,
               const Args&... args) {
  auto num_items = std::distance(s, e);
  auto* kernel =
      f.get_kernel(num_items); //(getCLContext()->get_default_device());
  kernel->set_work_size(num_items);
  (*kernel)();
}
} // namespace opencl
} // namespace galois

#endif /* GOPT_CL_HEADER_H_ */
