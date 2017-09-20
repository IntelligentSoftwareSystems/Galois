/**
 * @file
 * @section License
 *
 * This file is part of Galois.  Galois is a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 * @section Description
 *
 * @author Rashid Kaleem<rashid.kaleem@gmail.com>
 */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
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
#include "galois/OpenCL/CL_Errors.h"
#include "galois/OpenCL/CL_Util.h"
#include "galois/OpenCL/CL_Context.h"
#include "galois/OpenCL/CL_DeviceManager.h"
#include "galois/OpenCL/Arrays/Arrays.h"
#include "galois/OpenCL/graphs/Graphs.h"
//////////////////////////////////////////
#ifndef GOPT_CL_HEADER_H_
#define GOPT_CL_HEADER_H_
namespace galois{
namespace OpenCL{
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
