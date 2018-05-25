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

#ifndef GALOISGPU_OCL_CPUARRAY_H_
#define GALOISGPU_OCL_CPUARRAY_H_

namespace galois{
/*******************************************************************************
 *
 ********************************************************************************/
template<typename T>
struct CPUArray {
   typedef cl_mem DevicePtrType;
   typedef T * HostPtrType;
   explicit CPUArray(size_t sz, galois::opencl::CL_Device * d=nullptr) :
            num_elements(sz) {
      (void)d;
         host_data = new T[num_elements];
      }
      ////////////////////////////////////////////////
   void copy_to_device() {
   }
   ////////////////////////////////////////////////
   void copy_to_host() {
   }
   ////////////////////////////////////////////////
   void init_on_device(const T & val) {
      (void) val;
   }
   ////////////////////////////////////////////////
   size_t size() {
      return num_elements;
   }
   ////////////////////////////////////////////////
   operator T*() {
      return host_data;
   }
   T & operator [](size_t idx) {
      return host_data[idx];
   }
   DevicePtrType device_ptr(void) {
      return (DevicePtrType) 0;
   }
   HostPtrType & host_ptr(void) {
      return host_data;
   }
   CPUArray<T> * get_array_ptr(void) {
      return this;
   }
   ~CPUArray<T>() {
      if (host_data)
         delete[] host_data;
   }
   HostPtrType host_data;
   size_t num_elements;
protected:
};
}//end namespace galois



#endif /* GALOISGPU_OCL_CPUARRAY_H_ */
