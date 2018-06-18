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

#ifndef GALOISGPU_OCL_GPUARRAY_H_
#define GALOISGPU_OCL_GPUARRAY_H_

namespace galois {
namespace opencl {
/*******************************************************************************
 *
 ********************************************************************************/
template <typename T>
struct GPUArray {
  typedef cl_mem DevicePtrType;
  typedef T* HostPtrType;

  size_t num_elements;
  CL_Device* device;
  DevicePtrType device_data;
  int err;
  explicit GPUArray(size_t sz, CL_Device* d) : num_elements(sz), device(d) {
    DEBUG_CODE(fprintf(stderr,
                       "Allocating array %6.6g MB on device, total %6.6g\n",
                       (num_elements * sizeof(T) / (float)(1024 * 1024)),
                       (galois::opencl::OpenCL_Setup::allocated_bytes /
                        (float)(1024 * 1024)));)
    device_data = clCreateBuffer(device->context(), CL_MEM_READ_WRITE,
                                 sizeof(T) * num_elements, NULL, &err);
    ReportDataAllocation(device, sizeof(T) * num_elements);
  }
  ////////////////////////////////////////////////
  void copy_to_device(void* aux) {
    CHECK_CL_ERROR(err = clEnqueueWriteBuffer(
                       device->command_queue(), device_data, CL_TRUE, 0,
                       sizeof(T) * num_elements, (void*)(aux), 0, NULL, NULL),
                   " Copying aux to device ");
    ReportCopyToDevice(device, sizeof(T) * num_elements, err);
  }
  void copy_to_device(void* aux, size_t num_items) {
    CHECK_CL_ERROR(err = clEnqueueWriteBuffer(
                       device->command_queue(), device_data, CL_TRUE, 0,
                       sizeof(T) * num_items, (void*)(aux), 0, NULL, NULL),
                   " Copying aux to device ");
    ReportCopyToDevice(device, sizeof(T) * num_items, err);
  }
  ////////////////////////////////////////////////
  void copy_to_device(void* aux, cl_event* event) {
    CHECK_CL_ERROR(err = clEnqueueWriteBuffer(
                       device->command_queue(), device_data, CL_FALSE, 0,
                       sizeof(T) * num_elements, (void*)(aux), 0, NULL, event),
                   " Copying async aux to device ");
    ReportCopyToDevice(device, sizeof(T) * num_elements, err);
  }
  ////////////////////////////////////////////////
  void copy_to_host(void* host_ptr) {
    CHECK_CL_ERROR(err =
                       clEnqueueReadBuffer(device->command_queue(), device_data,
                                           CL_TRUE, 0, sizeof(T) * num_elements,
                                           (void*)(host_ptr), 0, NULL, NULL),
                   "Copying to host ");
    ReportCopyToHost(device, sizeof(T) * num_elements, err);
  }
  void copy_to_host(void* host_ptr, cl_event* event) {
    CHECK_CL_ERROR(err = clEnqueueReadBuffer(device->command_queue(),
                                             device_data, CL_FALSE, 0,
                                             sizeof(T) * num_elements,
                                             (void*)(host_ptr), 0, NULL, event),
                   "Copying to host ");
    ReportCopyToHost(device, sizeof(T) * num_elements, err);
  }
  void copy_to_host(void* host_ptr, size_t num_items) {
    CHECK_CL_ERROR(err = clEnqueueReadBuffer(
                       device->command_queue(), device_data, CL_TRUE, 0,
                       sizeof(T) * num_items, (void*)(host_ptr), 0, NULL, NULL),
                   "Copying to host ");
    ReportCopyToHost(device, sizeof(T) * num_items, err);
  }
  ////////////////////////////////////////////////
  size_t size() { return num_elements; }
  DevicePtrType& device_ptr(void) { return device_data; }
  DevicePtrType& device_ptr(CL_Device* d) {
    assert(d->id() == device->id() && "Invalid Device for this instance");
    return device_data;
  }
  GPUArray<T>* get_array_ptr(void) { return this; }
  ~GPUArray<T>() {
#ifdef _GOPT_DEBUG
    std::cout << "Deleting array device :: " << device_data << "\n";
#endif
    clReleaseMemObject(device_data);
    ReportDataAllocation(device, -sizeof(T) * num_elements);
  }

protected:
};
} // end namespace opencl
} // end namespace galois

#endif /* GALOISGPU_OCL_GPUARRAY_H_ */
