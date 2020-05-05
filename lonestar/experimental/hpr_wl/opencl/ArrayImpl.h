/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

/*
 * ArrayImpl.h
 *
 *  Created on: Apr 16, 2015
 *      Author: rashid
 */

#ifndef GALOISGPU_OCL_ARRAYIMPL_H_
#define GALOISGPU_OCL_ARRAYIMPL_H_

namespace galois {
namespace opencl {

/*******************************************************************************
 * TODO (RK) : Pinned memory should be fastest. Check if it is ok to replace all
 * instances of default Array<T> with the pinned version as default and update.
 ********************************************************************************/
extern CLEnvironment cl_env;
template <typename T>
struct Array {
  typedef cl_mem DevicePtrType;
  typedef T* HostPtrType;

  size_t num_elements;
  cl_mem_flags allocation_flags;
  HostPtrType host_data;
  DevicePtrType device_data;
  int err;

  explicit Array(unsigned long sz) : num_elements(sz) {

    host_data = nullptr;
    int ret =
        posix_memalign((void**)&host_data, 4096, sizeof(T) * num_elements);
    assert(ret == 0 && "Posix-memalign failed.");
    allocation_flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
    device_data      = clCreateBuffer(cl_env.m_context, allocation_flags,
                                 sizeof(T) * num_elements, host_data, &err);
    galois::opencl::CHECK_CL_ERROR(err, "Allocation failure...!");
  }
  ////////////////////////////////////////////////
  ~Array() {
#ifdef _GOPT_DEBUG
    std::cout << "Deleting array host:: " << host_data
              << " , device :: " << device_data << "\n";
#endif
    if (host_data) {
      free(host_data);
      host_data = nullptr;
      galois::opencl::CHECK_CL_ERROR(clReleaseMemObject(device_data),
                                     "Failed to release device memory object.");
    }
  }
  ////////////////////////////////////////////////
  void copy_to_device(size_t sz) {
    CHECK_CL_ERROR(err = clEnqueueWriteBuffer(
                       cl_env.m_command_queue, device_data, CL_TRUE, 0,
                       sizeof(T) * sz, (void*)(host_data), 0, NULL, NULL),
                   " Copying to device ");
  }

  void copy_to_device() {
    CHECK_CL_ERROR(
        err = clEnqueueWriteBuffer(cl_env.m_command_queue, device_data, CL_TRUE,
                                   0, sizeof(T) * num_elements,
                                   (void*)(host_data), 0, NULL, NULL),
        " Copying to device ");
  }
  ////////////////////////////////////////////////
  void copy_to_device(cl_event* event) {
    CHECK_CL_ERROR(
        err = clEnqueueWriteBuffer(cl_env.m_command_queue, device_data,
                                   CL_FALSE, 0, sizeof(T) * num_elements,
                                   (void*)(host_data), 0, NULL, event),
        " Copying async. to device ");
  }
  ////////////////////////////////////////////////
  void copy_to_device(void* aux) {
    CHECK_CL_ERROR(err = clEnqueueWriteBuffer(
                       cl_env.m_command_queue, device_data, CL_TRUE, 0,
                       sizeof(T) * num_elements, (void*)(aux), 0, NULL, NULL),
                   " Copying aux to device ");
  }
  ////////////////////////////////////////////////
  void copy_to_device(void* aux, size_t sz) {
    CHECK_CL_ERROR(err = clEnqueueWriteBuffer(
                       cl_env.m_command_queue, device_data, CL_TRUE, 0,
                       sizeof(T) * sz, (void*)(aux), 0, NULL, NULL),
                   " Copying aux to device ");
  }
  ////////////////////////////////////////////////

  void copy_to_host(size_t sz) {
    CHECK_CL_ERROR(err = clEnqueueReadBuffer(
                       cl_env.m_command_queue, device_data, CL_TRUE, 0,
                       sizeof(T) * sz, (void*)(host_data), 0, NULL, NULL),
                   "Copying to host ");
  }
  void copy_to_host(size_t sz, cl_event* event) {
    CHECK_CL_ERROR(err = clEnqueueReadBuffer(
                       cl_env.m_command_queue, device_data, CL_FALSE, 0,
                       sizeof(T) * sz, (void*)(host_data), 0, NULL, event),
                   "Copying to host ");
  }

  void copy_to_host() {
    CHECK_CL_ERROR(err =
                       clEnqueueReadBuffer(cl_env.m_command_queue, device_data,
                                           CL_TRUE, 0, sizeof(T) * num_elements,
                                           (void*)(host_data), 0, NULL, NULL),
                   "Copying to host ");
  }
  ////////////////////////////////////////////////
  size_t size() { return num_elements; }
  ////////////////////////////////////////////////
  operator T*() { return host_data; }
  T& operator[](size_t idx) { return host_data[idx]; }
  DevicePtrType& device_ptr(void) { return device_data; }
  HostPtrType& host_ptr(void) { return host_data; }
  Array<T>* get_array_ptr(void) { return this; }
  DevicePtrType get_sub_array(size_t start, size_t num_items) {
    cl_buffer_region buff;
    buff.origin = start * sizeof(T);
    buff.size   = num_items * sizeof(T);
    cl_int err;
    cl_mem res = clCreateSubBuffer(device_data, allocation_flags,
                                   CL_BUFFER_CREATE_TYPE_REGION, &buff, &err);
    galois::opencl::CHECK_CL_ERROR(err, "Failed to create sub-region!");
    return res;
  }

protected:
};

} // end namespace opencl
} // end namespace galois

#endif /* GALOISGPU_OCL_ARRAYIMPL_H_ */
