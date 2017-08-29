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
#ifndef GALOISGPU_OCL_CPUARRAY_H_
#define GALOISGPU_OCL_CPUARRAY_H_

namespace Galois{
/*******************************************************************************
 *
 ********************************************************************************/
template<typename T>
struct CPUArray {
   typedef cl_mem DevicePtrType;
   typedef T * HostPtrType;
   explicit CPUArray(size_t sz, Galois::OpenCL::CL_Device * d=nullptr) :
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
}//end namespace Galois



#endif /* GALOISGPU_OCL_CPUARRAY_H_ */
