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

#include "galois/opencl/CL_DeviceSet.h"
#ifndef GALOISGPU_OCL_MULTIDEVICEARRAY_H_
#define GALOISGPU_OCL_MULTIDEVICEARRAY_H_


namespace galois{
namespace opencl{
/*******************************************************************************
 *
 ********************************************************************************/
template<typename T>
struct MultiDeviceArray {

   typedef cl_mem DevicePtrType;
   typedef T * HostPtrType;
   typedef Array<T> ArrayType;

   size_t num_elements;
   HostPtrType host_data;
   std::vector<ArrayType *> data;


   explicit MultiDeviceArray(galois::opencl::DeviceSet * ds, unsigned long sz) :
           num_elements(sz) {
        host_data = new T[num_elements];
           for (auto d : ds->devices) {
              ArrayType *k = new ArrayType(sz,d);
              data.push_back(k);
           }
     }
   ////////////////////////////////////////////////
   void copy_to_device() {
      for(auto a : data){
         memcpy(a->host_ptr(),host_data,sizeof(T)*num_elements );
         a->copy_to_device();
      }
   }
   ////////////////////////////////////////////////
   void copy_to_device(cl_event * event) {
      for(auto a : data){
         memcpy(a->host_ptr(), host_data,sizeof(T)*num_elements );
                  a->copy_to_device();
      }
   }
   ////////////////////////////////////////////////
   void copy_to_device(void * aux) {
      memcpy(host_data, aux, sizeof(T)*num_elements);
      copy_to_device();
   }
   ////////////////////////////////////////////////
   void copy_to_host() {
      for(auto a : data){
         a->copy_to_host();
      }
   }
   ////////////////////////////////////////////////
   void init_on_device(const T & val) {
      for(auto a : data){
         a->init_on_device(val);
      }
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
   HostPtrType & host_ptr(void) {
      return host_data;
   }
   DevicePtrType & device_ptr(CL_Device * dev) {
      for(auto a : data){
         if(dev->id() == a->device->id()){
            return a->device_ptr();
         }
      }
      assert(false&&"Invalid device specified");
      return data[0]->device_ptr();//Error!
      }
   MultiDeviceArray<T> * get_array_ptr(void) {
      return this;
   }
   ~MultiDeviceArray<T>() {
#ifdef _GOPT_DEBUG
      std::cout<<"Deleting array host:: " << host_data << " , device :: " << device_data<<"\n";
#endif
      if (host_data){
         delete[] host_data;
         host_data=nullptr;
      }
      for(auto d : data)
         delete d;
   }
protected:
};

}//end namespace opencl
}//end namespace galois


#endif /* GALOISGPU_OCL_MULTIDEVICEARRAY_H_ */
