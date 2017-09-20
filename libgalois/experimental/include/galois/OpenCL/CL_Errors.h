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

#ifndef CL_ERROR_HANDLER_H_
#define CL_ERROR_HANDLER_H_
namespace galois {
namespace OpenCL {


#ifdef _GOPT_DEBUG
#define DEBUG_CODE(X) {X}
#define INFO_CODE(X) {X}
#else
#define DEBUG_CODE(X) {}
#define INFO_CODE(X) {}
#endif

/////////////////////////////////////////////////////////////////
inline const char* ocl_error_to_string(cl_int error) {
   switch (error) {
   case CL_SUCCESS:
      return "CL_SUCCESS";
   case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
   case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
   case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
   case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
   case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
   case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
   case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
   case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
   case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
   case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
   case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
   case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
   case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
   case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
   case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
   case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
   case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
   case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
   case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
   case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
   case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
   case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
   case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
   case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
   case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
   case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
   case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
   case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
   case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
   case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
   case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
   case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
   case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
   case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
   case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
   case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
   case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
   case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
   case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
   case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
   case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
   case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
   case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
   case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
   case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
   case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
      // unknown
   default:
      return "unknown error code";
   }
}
/////////////////////////////////////////////////////////////////
template<typename T>
void CHECK_ERROR_NULL(const T * obj, const char * const err_string) {
   if (obj == NULL) {
      std::cout << "Error occurred!! \"" << err_string << "\" Object is null!\n";
#ifndef _GOPT_DEBUG
#endif
      assert(false);
   }
   return;
}
/////////////////////////////////////////////////////////////////
template<typename T>
void CHECK_CL_ERROR(T err, const char * const err_string) {
   if (err != CL_SUCCESS) {
      std::cout << "Error occurred!! \"" << err_string << "\" code " << err << " " << ocl_error_to_string(err) << "\n";
#ifndef _GOPT_DEBUG
#endif
      assert(false);
   }
   return;
}
/////////////////////////////////////////////////////////////////
inline void check_command_queue(cl_command_queue & q) {
   std::cout << "\n" << "=======Begin queue check========\n";
   cl_context tmp_ctx;
   clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, 0, &tmp_ctx, NULL);
   std::cout << "Context " << tmp_ctx << "\n";
   cl_device_id dev_id;
   clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, 0, &dev_id, NULL);
   std::cout << "Device ID : " << dev_id << "\n";
   cl_uint count;
   clGetCommandQueueInfo(q, CL_QUEUE_REFERENCE_COUNT, 0, &count, NULL);
   std::cout << "Reference count :" << count << "\n";
   cl_command_queue_properties ppt;
   clGetCommandQueueInfo(q, CL_QUEUE_PROPERTIES, 0, &ppt, NULL);
   std::cout << "Properties :: ";
   if (ppt & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
      std::cout << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ";
   } else {
      std::cout << "IN_ORDER_EXEC_MODE_ ";
   }
   if (ppt & CL_QUEUE_PROFILING_ENABLE) {
      std::cout << "CL_QUEUE_PROFILING_ENABLE";
   } else {
      std::cout << "CL_QUEUE_PROFILING_DISABLED ";
   }
   std::cout << "\n" << "=======End queue check========\n";
   return;
}
/**********************************************************************
 *
 *
 **********************************************************************/
inline float toMB(long long b){
   return (float)(b)/(1024*1024);
}
inline void check_context(cl_context & ctx) {
   cl_uint ref_count;
   galois::OpenCL::CHECK_CL_ERROR(clGetContextInfo(ctx, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, 0), "Ref count failed");
   std::cout << "CheckCtx : RefCount[" << ref_count << "]";
   cl_device_id devices[10];
   size_t num_devs;
   galois::OpenCL::CHECK_CL_ERROR(clGetContextInfo(ctx, CL_CONTEXT_DEVICES, sizeof(cl_device_id) * 10, devices, &num_devs), "Dev count failed");
   std::cout << ", NumDev[" << num_devs << "]";
   cl_context_properties properties[10];
   size_t num_props;
   galois::OpenCL::CHECK_CL_ERROR(clGetContextInfo(ctx, CL_CONTEXT_PROPERTIES, sizeof(cl_context_properties) * 10, properties, &num_props), "Ref count failed");
   std::cout << ", NumProps[" << num_props << "], ";
}
/**********************************************************************
 *
 *
 **********************************************************************/

inline void check_device(cl_device_id p_id) {
   cl_ulong mem_cache_size;
   cl_device_mem_cache_type mem_type;
   cl_uint line_size;
   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_cache_size, NULL), "Global mem size");
   std::cout << "Global memory " << mem_cache_size << "\n";

   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &mem_cache_size, NULL), "Global mem cache size.");
   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cl_device_mem_cache_type), &mem_type, NULL), "Global mem cache type");
   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_ulong), &line_size, NULL), "Global mem cache line size");
   std::cout << "Global mem cache " << mem_cache_size << " , Line size " << line_size << " , ";
   if (mem_type & CL_READ_WRITE_CACHE)
      std::cout << "CL_READ_WRITE_CACHE ";
   if (mem_type & CL_READ_ONLY_CACHE)
      std::cout << "CL_READ_ONLY_CACHE";
   if (mem_type & CL_NONE)
      std::cout << "CL_NONE";
   std::cout << "\n";

   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_cache_size, NULL), "Local mem size");
   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_mem_cache_type), &mem_type, NULL), "Local mem type");
   std::cout << "Local mem cache " << mem_cache_size << " , ";
   if (mem_type & CL_READ_WRITE_CACHE)
      std::cout << "CL_READ_WRITE_CACHE ";
   if (mem_type & CL_READ_ONLY_CACHE)
      std::cout << "CL_READ_ONLY_CACHE";
   if (mem_type & CL_NONE)
      std::cout << "CL_NONE";
   std::cout << "\n";

   cl_ulong constant_size;
   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &constant_size, NULL), "Consant size");
   std::cout << "Constant buffer size " << (constant_size / (1024)) << " KB\n";
   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &line_size, NULL), "Max frequency");
   std::cout << "Max frequencey " << line_size << "\n";
   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &line_size, NULL), "Max compute units");
   std::cout << "Max Compute units " << line_size << "\n";
   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_cache_size, NULL), "Max mem allocation size");
   std::cout << "Max mem allocation size " << mem_cache_size << "\n";
   size_t temp;
   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &temp, NULL), "Max work-group size");
   std::cout << "Max work-group size " << temp << "\n";
   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &line_size, NULL), "Max work-item dimensions");
   std::cout << "Max work-item dimension " << line_size << "\n";
   size_t arr[10];
   CHECK_CL_ERROR(clGetDeviceInfo(p_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(arr), arr, NULL), "Max work-item size");
   std::cout << "Max work item sizes ";
   for (int i = 0; i < 10; ++i)
      std::cout << " " << arr[i] << ", ";
   std::cout << "\n";

}
/**********************************************************************
 *
 *
 **********************************************************************/
inline void print_info(const cl_device_id _device_id) {
   char string_holder[4 * 256];
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_NAME, 1024, string_holder, NULL), "clGetDeviceInfo-1");
   char cl_version[256];
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_VERSION, 256, cl_version, NULL), "clGetDeviceInfo-2");
   char cl_c_version[256];
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_OPENCL_C_VERSION, 256, cl_c_version, NULL), "clGetDeviceInfo-3");
   cl_bool use_unified_mem;
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &use_unified_mem, NULL), "clGetDeviceInfo-4");
   cl_uint freq;
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &freq, NULL), "Max frequency");
   size_t max_wg_size;
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL), "clGetDeviceInfo");
   cl_uint num_eus;
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_eus, NULL), "clGetDeviceInfo");
   cl_ulong global_mem_size;
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL), "Global mem size");
   cl_ulong max_alloc_size;
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc_size, NULL), "Global mem size");
   cl_uint mem_align_size;
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &mem_align_size, NULL), "Mem alignment(bits)");
   cl_uint mem_address_bits;
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &mem_address_bits, NULL), "Mem address(bits)");
   std::cerr << "" << string_holder << " @" << freq << " Hz, [" << num_eus << " EUs, (max-workgroup=" << max_wg_size << ")" << toMB(global_mem_size) << "MB (Max-"
         << toMB(max_alloc_size) << " MB " << mem_align_size << "-bit aligned) Address:" << mem_address_bits << "-bits] (CL::" << cl_version << "CL_CC:: " << cl_c_version << ")";
#if 0//ifdef CL_VERSION_1_2 //Device affinity only supported in opencl 1.2
   cl_device_affinity_domain affinity;
   CHECK_CL_ERROR(clGetDeviceInfo(_device_id, CL_DEVICE_PARTITION_AFFINITY_DOMAIN, sizeof(cl_device_affinity_domain), &affinity, NULL), "Global mem size");
   std::cout<<" :: Affinity ::"<< affinity;
#endif
   std::cerr << ", Unified Mem:" << (use_unified_mem ? "Y" : "N") << "\n";
}
inline cl_uint get_device_eu(cl_device_id dev) {
   cl_uint num_eus;
   CHECK_CL_ERROR(clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_eus, NULL), "clGetDeviceInfo");
   return num_eus;
}
inline cl_uint get_device_threads(cl_device_id dev) {
   cl_uint num_eus;
   CHECK_CL_ERROR(clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_eus, NULL), "clGetDeviceInfo");
   size_t max_wg_size;
   CHECK_CL_ERROR(clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL), "clGetDeviceInfo");
   return num_eus * max_wg_size;
}
inline cl_ulong get_device_memory(cl_device_id dev) {
   cl_ulong ret;
   CHECK_CL_ERROR(clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &ret, NULL), "Global mem size");
   return ret;
}
inline cl_ulong get_device_shared_memory(cl_device_id dev) {
   cl_ulong ret;
   CHECK_CL_ERROR(clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &ret, NULL), "Local mem size");
   return ret;
}

inline cl_ulong get_max_allocation_size(cl_device_id dev) {
   cl_ulong ret;
   CHECK_CL_ERROR(clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &ret, NULL), "Global mem size");
   return ret;
}
}
}

#endif /* CL_ERROR_HANDLER_H_ */
