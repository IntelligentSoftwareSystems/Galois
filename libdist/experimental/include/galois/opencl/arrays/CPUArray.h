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
