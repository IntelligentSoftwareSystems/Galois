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

#ifndef ARRAY_WRAPPER_H_
#define ARRAY_WRAPPER_H_

namespace galois {
namespace opencl {

/////////////////////////////////////////////////////////////////
/*******************************************************************************
 *
 ********************************************************************************/
enum DeviceMemoryType { DISCRETE, HOST_CACHED, PINNED, CONSTANT };
static inline void ReportCopyToDevice(CL_Device* dev, size_t sz,
                                      cl_int err = CL_SUCCESS) {
  dev->stats().copied_to_device += sz;
  (void)err;
  DEBUG_CODE(if (err != CL_SUCCESS)
                 fprintf(stderr, "Failed copy to device [ %d bytes ]!\n", sz);
             else fprintf(stderr, "Did copy to device[ %d bytes ] !\n", sz);)
}
/////////////////////////////////
static inline void ReportCopyToHost(CL_Device* dev, size_t sz,
                                    cl_int err = CL_SUCCESS) {
  dev->stats().copied_to_host += sz;
  (void)err;
  DEBUG_CODE(if (err != CL_SUCCESS)
                 fprintf(stderr, "Failed copy to host [ %d bytes ]!\n", sz);
             else fprintf(stderr, "Did copy to device[ %d host ] !\n", sz);)
}
/////////////////////////////////////
static inline void ReportDataAllocation(CL_Device* dev, size_t sz,
                                        cl_int err = CL_SUCCESS) {
  dev->stats().allocated += sz;
  dev->stats().max_allocated =
      std::max(dev->stats().max_allocated, dev->stats().allocated);
  DEBUG_CODE(fprintf(stderr, "Allocating array %6.6g MB on device-%d (%s)\n",
                     (sz / (float)(1024 * 1024)), dev->id(),
                     dev->name().c_str());
             dev->stats().print_long();)
}

} // namespace opencl

/*******************************************************************************
 *
 ********************************************************************************/
} // namespace galois
#include "galois/opencl/arraysArrayImpl.h"
#include "galois/opencl/arraysMultiDeviceArray.h"
#include "galois/opencl/arraysCPUArray.h"
#include "galois/opencl/arraysGPUArray.h"
#include "galois/opencl/arraysOnDemandArray.h"
#endif /* ARRAY_WRAPPER_H_ */
