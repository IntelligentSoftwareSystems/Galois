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
#include "Galois/OpenCL/CL_DeviceSet.h"
#ifndef GALOISGPU_OCL_MULTIDEVICEARRAY_H_
#define GALOISGPU_OCL_MULTIDEVICEARRAY_H_


namespace Galois{
namespace OpenCL{
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


   explicit MultiDeviceArray(Galois::OpenCL::DeviceSet * ds, unsigned long sz) :
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

}//end namespace OpenCL
}//end namespace Galois


#endif /* GALOISGPU_OCL_MULTIDEVICEARRAY_H_ */
