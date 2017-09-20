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
#ifndef ARRAY_WRAPPER_H_
#define ARRAY_WRAPPER_H_

namespace galois {
namespace opencl {

/////////////////////////////////////////////////////////////////
/*******************************************************************************
 *
 ********************************************************************************/
enum DeviceMemoryType {
   DISCRETE, HOST_CACHED, PINNED, CONSTANT
};
static inline void ReportCopyToDevice(CL_Device * dev, size_t sz, cl_int err = CL_SUCCESS) {
   dev->stats().copied_to_device += sz;
   (void) err;
DEBUG_CODE(
      if(err!=CL_SUCCESS)fprintf(stderr, "Failed copy to device [ %d bytes ]!\n", sz);
      else fprintf(stderr, "Did copy to device[ %d bytes ] !\n", sz);)
}
/////////////////////////////////
static inline void ReportCopyToHost(CL_Device * dev, size_t sz, cl_int err = CL_SUCCESS) {
dev->stats().copied_to_host += sz;
(void) err;
DEBUG_CODE(
   if(err!=CL_SUCCESS)fprintf(stderr, "Failed copy to host [ %d bytes ]!\n", sz);
   else fprintf(stderr, "Did copy to device[ %d host ] !\n", sz);)
}
/////////////////////////////////////
static inline void ReportDataAllocation(CL_Device * dev, size_t sz, cl_int err = CL_SUCCESS) {
dev->stats().allocated += sz;
dev->stats().max_allocated = std::max(dev->stats().max_allocated, dev->stats().allocated);
DEBUG_CODE(
   fprintf(stderr, "Allocating array %6.6g MB on device-%d (%s)\n", (sz / (float) (1024 * 1024)), dev->id(), dev->name().c_str());
   dev->stats().print_long();
)
}

}      //namespace opencl

/*******************************************************************************
 *
 ********************************************************************************/
}      //namespace galois
#include "galois/opencl/Arrays/ArrayImpl.h"
#include "galois/opencl/Arrays/MultiDeviceArray.h"
#include "galois/opencl/Arrays/CPUArray.h"
#include "galois/opencl/Arrays/GPUArray.h"
#include "galois/opencl/Arrays/OnDemandArray.h"
#endif /* ARRAY_WRAPPER_H_ */
