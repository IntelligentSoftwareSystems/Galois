/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
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

#ifndef GALOISGPU_OCL_CLKERNEL_H_
#define GALOISGPU_OCL_CLKERNEL_H_

namespace galois {
namespace opencl {
/*************************************************************************
 *
 *************************************************************************/
struct CL_Kernel {
  cl_kernel kernel;
  CL_Device* device;
  size_t global, local;
  double total_time;
  cl_event event;
  char name[256];
  /*************************************************************************
   *
   *************************************************************************/
  explicit CL_Kernel(CL_Device* d) : device(d), total_time(0), event(0) {
    global = local = 0;
    kernel         = 0;
  } /*************************************************************************
     *
     *************************************************************************/
  explicit CL_Kernel() : device(0), total_time(0), event(0) {
    global = local = 0;
    kernel         = 0;
  }
  /*************************************************************************
   *
   *************************************************************************/
  CL_Kernel(CL_Device* d, const char* filename, const char* kernel_name,
            bool from_string)
      : device(d), total_time(0), event(0) {
    if (from_string) {
      init_string(filename, kernel_name);
    } else {
      init(filename, kernel_name);
    }
  }
  /*************************************************************************
   *
   *************************************************************************/
  ~CL_Kernel() {
    if (kernel) {
      clReleaseKernel(kernel);
    }
  }
  /*************************************************************************
   *
   *************************************************************************/
  void init(CL_Device* d, const char* filename, const char* kernel_name) {
    device = d;
    strcpy(name, kernel_name);
    global = local = 0; // galois::opencl::OpenCL_Setup::get_default_device();
    kernel = galois::opencl::getCLContext()->load_kernel(filename, kernel_name,
                                                         device);
  }
  /*************************************************************************
   *
   *************************************************************************/
  void init(const char* filename, const char* kernel_name) {
    strcpy(name, kernel_name);
    global = local = 0; // galois::opencl::OpenCL_Setup::get_default_device();
    kernel = galois::opencl::getCLContext()->load_kernel(filename, kernel_name,
                                                         device);
  }
  /*************************************************************************
   *
   *************************************************************************/
  void init_string(const char* src, const char* kernel_name) {
    strcpy(name, kernel_name);
    global = local = 0;
    kernel         = galois::opencl::getCLContext()->load_kernel_string(
        src, kernel_name, device);
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_work_size(size_t num_items) {
    local  = galois::opencl::getCLContext()->workgroup_size(kernel, device);
    global = (size_t)(ceil(num_items / ((double)local)) * local);
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_work_size(size_t num_items, size_t local_size) {
    local  = local_size;
    global = (size_t)(ceil(num_items / ((double)local)) * local);
  }
  /*************************************************************************
   *
   *************************************************************************/
  size_t get_default_workgroup_size() {
    return galois::opencl::getCLContext()->workgroup_size(kernel, device);
  }
  /*************************************************************************
   *
   *************************************************************************/
  cl_ulong get_shared_memory_usage() {
    cl_ulong mem_usage;
    CHECK_CL_ERROR(clGetKernelWorkGroupInfo(kernel, device->id(),
                                            CL_KERNEL_LOCAL_MEM_SIZE,
                                            sizeof(cl_ulong), &mem_usage, NULL),
                   "Error: Failed to get shared memory usage for kernel.");
    return mem_usage;
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg(unsigned int index, size_t sz, const void* val) {
    galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, index, sz, val),
                                   "Arg, compact is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg_shmem(unsigned int index, size_t sz) {
    galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, index, sz, nullptr),
                                   "Arg, shared-mem is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg_max_shmem(unsigned int index) {
    size_t sz = device->get_device_shared_memory() - get_shared_memory_usage();
    galois::opencl::CHECK_CL_ERROR(clSetKernelArg(kernel, index, sz, nullptr),
                                   "Arg, max-shared-mem is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  size_t get_avail_shmem() {
    size_t sz = device->get_device_shared_memory() - get_shared_memory_usage();
    return sz;
  }
  /*************************************************************************
   *
   *************************************************************************/
  template <typename T>
  void set_arg(unsigned int index, const T& val) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, index, sizeof(cl_mem), &val->device_ptr()),
        "Arg, compact is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  template <typename T>
  void set_arg_list(const T& val) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val->device_ptr()),
        "Arg-1, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  template <typename T1, typename T2>
  void set_arg_list(const T1& val1, const T2& val2) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()),
        "Arg-1/2, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()),
        "Arg-2/2, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  template <typename T1, typename T2, typename T3>
  void set_arg_list(const T1& val1, const T2& val2, const T3& val3) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()),
        "Arg-1/3, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()),
        "Arg-2/3, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()),
        "Arg-3/3, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  template <typename T1, typename T2, typename T3, typename T4>
  void set_arg_list(const T1& val1, const T2& val2, const T3& val3,
                    const T4& val4) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()),
        "Arg-1/4, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()),
        "Arg-2/4, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()),
        "Arg-3/4, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4->device_ptr()),
        "Arg-4/4, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  void set_arg_list(const T1& val1, const T2& val2, const T3& val3,
                    const T4& val4, const T5& val5) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()),
        "Arg-1/5, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()),
        "Arg-2/5, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()),
        "Arg-3/5, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4->device_ptr()),
        "Arg-4/5, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5->device_ptr()),
        "Arg-5/5, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6>
  void set_arg_list(const T1& val1, const T2& val2, const T3& val3,
                    const T4& val4, const T5& val5, const T6& val6) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()),
        "Arg-1/6, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()),
        "Arg-2/6, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()),
        "Arg-3/6, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4->device_ptr()),
        "Arg-4/6, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5->device_ptr()),
        "Arg-5/6, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &val6->device_ptr()),
        "Arg-6/6, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7>
  void set_arg_list(const T1& val1, const T2& val2, const T3& val3,
                    const T4& val4, const T5& val5, const T6& val6,
                    const T7& val7) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()),
        "Arg-1/7, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()),
        "Arg-2/7, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()),
        "Arg-3/7, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4->device_ptr()),
        "Arg-4/7, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5->device_ptr()),
        "Arg-5/7, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &val6->device_ptr()),
        "Arg-6/7, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &val7->device_ptr()),
        "Arg-7/7, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
            typename T6, typename T7, typename T8>
  void set_arg_list(const T1& val1, const T2& val2, const T3& val3,
                    const T4& val4, const T5& val5, const T6& val6,
                    const T7& val7, const T8& val8) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1->device_ptr()),
        "Arg-1/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2->device_ptr()),
        "Arg-2/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3->device_ptr()),
        "Arg-3/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4->device_ptr()),
        "Arg-4/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5->device_ptr()),
        "Arg-5/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &val6->device_ptr()),
        "Arg-6/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &val7->device_ptr()),
        "Arg-7/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 7, sizeof(cl_mem), &val8->device_ptr()),
        "Arg-8/8, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg_raw(size_t idx, const cl_mem val) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, idx, sizeof(cl_mem), &val),
        "Arg-index, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg_list_raw(const cl_mem val) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val), "Arg-1, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg_list_raw(const cl_mem& val1, const cl_mem& val2) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1),
        "Arg-1/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2),
        "Arg-2/8, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg_list_raw(const cl_mem& val1, const cl_mem& val2,
                        const cl_mem& val3) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1),
        "Arg-1/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2),
        "Arg-2/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3),
        "Arg-3/8, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg_list_raw(const cl_mem& val1, const cl_mem& val2,
                        const cl_mem& val3, const cl_mem& val4) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1),
        "Arg-1/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2),
        "Arg-2/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3),
        "Arg-3/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4),
        "Arg-4/8, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg_list_raw(const cl_mem& val1, const cl_mem& val2,
                        const cl_mem& val3, const cl_mem& val4,
                        const cl_mem& val5) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1),
        "Arg-1/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2),
        "Arg-2/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3),
        "Arg-3/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4),
        "Arg-4/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5),
        "Arg-5/8, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg_list_raw(const cl_mem& val1, const cl_mem& val2,
                        const cl_mem& val3, const cl_mem& val4,
                        const cl_mem& val5, const cl_mem& val6) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1),
        "Arg-1/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2),
        "Arg-2/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3),
        "Arg-3/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4),
        "Arg-4/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5),
        "Arg-5/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &val6),
        "Arg-6/8, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg_list_raw(const cl_mem& val1, const cl_mem& val2,
                        const cl_mem& val3, const cl_mem& val4,
                        const cl_mem& val5, const cl_mem& val6,
                        const cl_mem& val7) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1),
        "Arg-1/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2),
        "Arg-2/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3),
        "Arg-3/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4),
        "Arg-4/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5),
        "Arg-5/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &val6),
        "Arg-6/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &val7),
        "Arg-7/8, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void set_arg_list_raw(const cl_mem& val1, const cl_mem& val2,
                        const cl_mem& val3, const cl_mem& val4,
                        const cl_mem& val5, const cl_mem& val6,
                        const cl_mem& val7, const cl_mem& val8) {
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &val1),
        "Arg-1/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &val2),
        "Arg-2/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &val3),
        "Arg-3/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &val4),
        "Arg-4/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &val5),
        "Arg-5/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &val6),
        "Arg-6/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &val7),
        "Arg-7/8, is NOT set!");
    galois::opencl::CHECK_CL_ERROR(
        clSetKernelArg(kernel, 7, sizeof(cl_mem), &val8),
        "Arg-8/8, is NOT set!");
  }
  /*************************************************************************
   *
   *************************************************************************/
  double get_kernel_time() { return total_time; }
  /*************************************************************************
   *
   *************************************************************************/
  void set_max_threads() {
    set_work_size(local * galois::opencl::getCLContext()->get_device_eu());
  }
  /*************************************************************************
   *
   *************************************************************************/
  void operator()() {
    //      fprintf(stderr, "Launching kernel [%s] [%u,%u]\n", name, local,
    //      global);
    galois::opencl::CHECK_CL_ERROR(
        clEnqueueNDRangeKernel(device->command_queue(), kernel, 1, nullptr,
                               &global, &local, 0, nullptr, &event),
        name);
  }
  /*************************************************************************
   *
   *************************************************************************/
  void run_task() {
    galois::opencl::CHECK_CL_ERROR(
        clEnqueueTask(device->command_queue(), kernel, 0, nullptr, &event),
        name);
  }
  /*************************************************************************
   *
   *************************************************************************/
  void operator()(cl_event& e) {
    galois::opencl::CHECK_CL_ERROR(
        clEnqueueNDRangeKernel(device->command_queue(), kernel, 1, NULL,
                               &global, &local, 0, NULL, &e),
        name);
  }
  /*************************************************************************
   *
   *************************************************************************/
  void wait() {
    galois::opencl::CHECK_CL_ERROR(clWaitForEvents(1, &event),
                                   "Error in waiting for kernel.");
  }
  /*************************************************************************
   *
   *************************************************************************/
  void operator()(size_t num_events, cl_event* events) {
    galois::opencl::CHECK_CL_ERROR(
        clEnqueueNDRangeKernel(device->command_queue(), kernel, 1, NULL,
                               &global, &local, num_events, events, &event),
        name);
  }
  /*************************************************************************
   *
   *************************************************************************/
  float last_exec_time() {
#ifdef _GOPT_CL_ENABLE_PROFILING_
    cl_ulong start_time, end_time;
    galois::opencl::CHECK_CL_ERROR(clWaitForEvents(1, &event),
                                   "Waiting for kernel");
    galois::opencl::CHECK_CL_ERROR(
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong), &start_time, NULL),
        "Kernel start time failed.");
    galois::opencl::CHECK_CL_ERROR(
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong), &end_time, NULL),
        "Kernel end time failed.");
    return (end_time - start_time) / (float)1.0e9f;
#else
    return 0.0f;
#endif
  }
  /*************************************************************************
   *
   *************************************************************************/
};
} // namespace opencl
} // namespace galois

#endif /* GALOISGPU_OCL_CLKERNEL_H_ */
