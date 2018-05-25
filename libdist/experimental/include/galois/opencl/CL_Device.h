/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#include "CL_Errors.h"
#ifndef GALOISGPU_OCL_CL_DEVICE_H_
#define GALOISGPU_OCL_CL_DEVICE_H_

namespace galois{
namespace opencl{

struct CL_Device {
protected:
   friend struct galois::opencl::CL_Platform;
   cl_device_id _id;
   cl_command_queue _queue;
   cl_context _context;
   const CL_Platform * platform;
#if _GALOIS_BUILD_INITIALIZER_KERNEL_
   cl_kernel init_kernel;
   const char * init_kernel_str; //= ;
#endif
   DeviceStats _stats;
   explicit CL_Device(CL_Platform * p, cl_device_id p_id) :
         _id(0),platform(p)
#if _GALOIS_BUILD_INITIALIZER_KERNEL_
   , init_kernel_str("__kernel void init(__global int * arr, int size, int val){const int id = get_global_id(0); if(id < size){arr[id]=val;}}")
#endif
   {
      initialize(p_id);
      // Create the compute program from the source buffer                   [6]
#if _GALOIS_BUILD_INITIALIZER_KERNEL_
      cl_int err;
      cl_program init_program = clCreateProgramWithSource(_context, 1, &init_kernel_str, NULL, &err);
      CHECK_CL_ERROR(err, "clCreateProgramWithSource failed.");
      CHECK_ERROR_NULL(init_program, "clCreateProgramWithSource");
      // Build the program executable                                        [7]
      err = clBuildProgram(init_program, 0, NULL, "", NULL, NULL);
      if (err != CL_SUCCESS) {
         size_t len;
         char buffer[10 * 2048];
         clGetProgramBuildInfo(init_program, _id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
         std::cout << "\n====Kernel build log====\n" << buffer << "\n=====END LOG=====\n";
      }
      init_kernel = clCreateKernel(init_program, "init", &err);
#endif
   }
public:
   void finish(){
      clFinish(command_queue());
   }
   cl_device_id id(){
      return _id;
   }
   cl_command_queue command_queue(){
      return _queue;
   }
   cl_context context(){
      return _context;
   }
   void print_stats() {
      fprintf(stderr, "Device: %s, ", name().c_str());
      _stats.print();
      fprintf(stderr, "\n");
   }
   void print() const {
      print_info(_id);
   }
   DeviceStats & stats(){
      return _stats;
   }
   void build_init_kernel();
   template<typename T>
   void init_on_device(cl_mem arr, size_t sz, const T & val);
   static float toMB(long v) {
      return v / (float) (1024 * 1024);
   }
   ~CL_Device() {
      fprintf(stderr, "Released device :: %s \n", name().c_str());
   }
   const CL_Platform* get_platform() {
//      fprintf(stderr, "Getting platform for %s\n", name().c_str());
      return platform;
   }
   cl_ulong get_device_shared_memory() {
      cl_ulong ret;
      CHECK_CL_ERROR(clGetDeviceInfo(_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &ret, NULL), "Local mem size");
      return ret;
   }
   cl_ulong get_max_allocation_size() {
      cl_ulong ret;
      CHECK_CL_ERROR(clGetDeviceInfo(_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &ret, NULL), "Global mem size");
      return ret;
   }
   cl_uint get_device_threads() {
      cl_uint num_eus;
      CHECK_CL_ERROR(clGetDeviceInfo(_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_eus, NULL), "clGetDeviceInfo");
      size_t max_wg_size;
      CHECK_CL_ERROR(clGetDeviceInfo(_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL), "clGetDeviceInfo");
      return num_eus * max_wg_size;
   }
   void initialize(cl_device_id p_id) {
      _id = p_id;
//      check_device(_id);
      cl_int err;
      _context = clCreateContext(0, 1, &_id, NULL, NULL, &err);
      CHECK_ERROR_NULL(&_context, "clCreateContext");
      //CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
#ifdef _GOPT_CL_ENABLE_PROFILING_
      _queue = clCreateCommandQueue(_context, _id, CL_QUEUE_PROFILING_ENABLE, &err);
#else
      _queue = clCreateCommandQueue(_context, _id, 0 /*CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/, &err);
#endif
      CHECK_ERROR_NULL(&_queue, "clCreateCommandQueue ");
   DEBUG_CODE(check_command_queue(_queue);)
//      print();
}


std::string name() const {
//   check_device(id);
   char string_holder[4 * 256];
   CHECK_CL_ERROR(clGetDeviceInfo(_id, CL_DEVICE_NAME, 1024, string_holder, NULL), "clGetDeviceInfo-1");
   return std::string(string_holder);
}


};
}
}



#endif /* GALOISGPU_OCL_CL_DEVICE_H_ */
