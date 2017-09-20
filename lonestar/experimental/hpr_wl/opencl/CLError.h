/*
 * CLError.h
 *
 *  Created on: Jun 26, 2015
 *      Author: rashid
 */

#ifndef GDIST_EXP_APPS_HPR_CL_CLERROR_H_
#define GDIST_EXP_APPS_HPR_CL_CLERROR_H_


/*
 * cl_error_handler.h
 *
 *  Created on: Nov 26, 2013
 *      Author: rashid
 */
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

#ifndef CL_ERROR_HANDLER_H_
#define CL_ERROR_HANDLER_H_
namespace galois {
namespace opencl {
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
}
}

#endif /* CL_ERROR_HANDLER_H_ */



#endif /* GDIST_EXP_APPS_HPR_CL_CLERROR_H_ */
