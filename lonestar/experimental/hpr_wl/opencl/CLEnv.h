/*
 * CLEnv.h
 *
 *  Created on: Jun 30, 2015
 *      Author: rashid
 */

#ifndef GDIST_EXP_APPS_HPR_OPENCL_CLENV_H_
#define GDIST_EXP_APPS_HPR_OPENCL_CLENV_H_

extern "C" {
#include <CL/cl.h>
}
#include "CLError.h"

//#define DEFAULT_CL_PLATFORM_ID 0
//#define DEFAULT_CL_DEVICE_ID 0
namespace galois {
namespace opencl {
struct CLEnvironment {
  std::string vendor_name;
  cl_platform_id m_platform_id;
  cl_device_id m_device_id;
  cl_context m_context;
  cl_command_queue m_command_queue;
  cl_program m_program;

  CLEnvironment()
      : m_platform_id(0), m_device_id(0), m_context(0), m_command_queue(0),
        m_program(0) {}
  void init(float dev_id) {
    setenv("CUDA_CACHE_DISABLE", "1", 1);
    const int DEFAULT_CL_PLATFORM_ID = (int)(dev_id) % 10;
    const int DEFAULT_CL_DEVICE_ID   = (int)((dev_id * 10)) % 10;
    char string_holder[4 * 256];
    cl_platform_id l_platforms[4];
    cl_uint num_platforms;
    m_program = nullptr;
    CHECK_CL_ERROR(clGetPlatformIDs(4, l_platforms, &num_platforms),
                   "clGetPlatformIDs ");
    m_platform_id = l_platforms[DEFAULT_CL_PLATFORM_ID];
    clGetPlatformInfo(l_platforms[DEFAULT_CL_PLATFORM_ID], CL_PLATFORM_NAME,
                      256, string_holder, NULL);
    std::cout << "Platform: " << string_holder;
    clGetPlatformInfo(l_platforms[DEFAULT_CL_PLATFORM_ID], CL_DEVICE_VENDOR,
                      256, string_holder, NULL);
    std::cout << ", Vendor: " << string_holder;
    vendor_name = string_holder;
    CHECK_CL_ERROR(clGetPlatformInfo(l_platforms[DEFAULT_CL_PLATFORM_ID],
                                     CL_PLATFORM_VERSION, 256, string_holder,
                                     NULL),
                   "clGetPlatform info, version");
    std::cout << "[Version: [" << string_holder << "] ";
    cl_uint num_devices;
    clGetDeviceIDs(l_platforms[DEFAULT_CL_PLATFORM_ID], CL_DEVICE_TYPE_ALL, 0,
                   0, &num_devices);
    cl_device_id* l_devices = new cl_device_id[num_devices];
    clGetDeviceIDs(l_platforms[DEFAULT_CL_PLATFORM_ID], CL_DEVICE_TYPE_ALL,
                   num_devices, l_devices, 0);
    std::cout << ",devices=" << num_devices << " \n";
    m_device_id = l_devices[DEFAULT_CL_DEVICE_ID];
    cl_int err;
    m_context = clCreateContext(0, 1, &m_device_id, NULL, NULL, &err);
    CHECK_ERROR_NULL(&m_context, "clCreateContext");
    // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
#ifdef _GOPT_CL_ENABLE_PROFILING_
    _queue =
        clCreateCommandQueue(_context, _id, CL_QUEUE_PROFILING_ENABLE, &err);
#else
    m_command_queue = clCreateCommandQueue(
        m_context, m_device_id, 0 /*CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/,
        &err);
#endif

    delete[] l_devices;
  }
  ~CLEnvironment() {
    clReleaseCommandQueue(m_command_queue);
    clReleaseContext(m_context);
    //      clReleaseDevice(m_device_id);
  }
} cl_env; // CLEnvironment
} // namespace opencl
} // namespace galois

#endif /* GDIST_EXP_APPS_HPR_OPENCL_CLENV_H_ */
