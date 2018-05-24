#ifndef GALOISGPU_OCL_ARRAYIMPL_H_
#define GALOISGPU_OCL_ARRAYIMPL_H_

namespace galois {
namespace opencl {

/*******************************************************************************
 * TODO (RK) : Pinned memory should be fastest. Check if it is ok to replace all
 * instances of default Array<T> with the pinned version as default and update.
 ********************************************************************************/
template<typename T>
struct Array {
   typedef cl_mem DevicePtrType;
   typedef T * HostPtrType;

   size_t num_elements;
   CL_Device * device;
   cl_mem_flags allocation_flags;
   HostPtrType host_data;
   DevicePtrType device_data;
   int err;

   explicit Array(unsigned long sz, CL_Device * _d , DeviceMemoryType MemType = DISCRETE) :
         num_elements(sz), device(_d) {
      switch (MemType) {
      case HOST_CACHED:
         allocation_flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
         break;
      case PINNED:
         allocation_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
         break;
      case CONSTANT:
         allocation_flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
         break;
      case DISCRETE:
      default:
         allocation_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
         break;
      }
#if 0
      allocation_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
      device_data = clCreateBuffer(device->context(), allocation_flags, sizeof(T) * num_elements, NULL, &err);
      galois::opencl::CHECK_CL_ERROR(err, "Allocation failure...!");
      host_data = (T*) clEnqueueMapBuffer(device->command_queue(), device_data, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(T) * num_elements, 0, NULL, NULL, &err);
#else
      host_data=  nullptr;
      int ret = posix_memalign((void **)&host_data, 4096, sizeof(T)*num_elements);
      assert(ret==0 && "Posix-memalign failed." );
      if(ret!=0){
         fprintf(stderr, "posix memallocation failed[%d]!", ret);
         exit(-1);
      }
      allocation_flags = CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR;
      device_data = clCreateBuffer(device->context(), allocation_flags, sizeof(T) * num_elements, host_data, &err);
      galois::opencl::CHECK_CL_ERROR(err, "Allocation failure...!");
#endif
      ReportDataAllocation(device, sizeof(T) * num_elements, err);
   }
   ////////////////////////////////////////////////
   ~Array(){
#ifdef _GOPT_DEBUG
      std::cout<<"Deleting array host:: " << host_data << " , device :: " << device_data<<"\n";
#endif
      if(host_data){
         free(host_data);
         host_data = nullptr;
         galois::opencl::CHECK_CL_ERROR(clReleaseMemObject(device_data),"Failed to release device memory object.");
         ReportDataAllocation(device, -1 * sizeof(T) * num_elements);
      }
   }
   ////////////////////////////////////////////////
   void copy_to_device(size_t sz) {
      CHECK_CL_ERROR(err = clEnqueueWriteBuffer(device->command_queue(), device_data, CL_TRUE, 0, sizeof(T) * sz, (void*) (host_data), 0, NULL, NULL), " Copying to device ");
      ReportCopyToDevice(device, sizeof(T) * sz, err);
   }

   void copy_to_device() {
      CHECK_CL_ERROR(err = clEnqueueWriteBuffer(device->command_queue(), device_data, CL_TRUE, 0, sizeof(T) * num_elements, (void*) (host_data), 0, NULL, NULL),
            " Copying to device ");
      ReportCopyToDevice(device, sizeof(T) * num_elements, err);
   }
   ////////////////////////////////////////////////
   void copy_to_device(cl_event * event) {
      CHECK_CL_ERROR(err = clEnqueueWriteBuffer(device->command_queue(), device_data, CL_FALSE, 0, sizeof(T) * num_elements, (void*) (host_data), 0, NULL, event),
            " Copying async. to device ");
      ReportCopyToDevice(device, sizeof(T) * num_elements, err);
   }
   ////////////////////////////////////////////////
   void copy_to_device(void * aux) {
      CHECK_CL_ERROR(err = clEnqueueWriteBuffer(device->command_queue(), device_data, CL_TRUE, 0, sizeof(T) * num_elements, (void*) (aux), 0, NULL, NULL),
            " Copying aux to device ");
      ReportCopyToDevice(device, sizeof(T) * num_elements, err);
   }
   ////////////////////////////////////////////////
   void copy_to_device(void * aux, size_t sz) {
      CHECK_CL_ERROR(err = clEnqueueWriteBuffer(device->command_queue(), device_data, CL_TRUE, 0, sizeof(T) * sz, (void*) (aux), 0, NULL, NULL), " Copying aux to device ");
      ReportCopyToDevice(device, sizeof(T) * sz, err);
   }
   ////////////////////////////////////////////////

   void copy_to_host(size_t sz) {
      CHECK_CL_ERROR(err = clEnqueueReadBuffer(device->command_queue(), device_data, CL_TRUE, 0, sizeof(T) * sz, (void*) (host_data), 0, NULL, NULL), "Copying to host ");
      ReportCopyToHost(device, sizeof(T) * sz, err);
   }
   void copy_to_host(size_t sz, cl_event *event) {
      CHECK_CL_ERROR(err = clEnqueueReadBuffer(device->command_queue(), device_data, CL_FALSE, 0, sizeof(T) * sz, (void*) (host_data), 0, NULL, event), "Copying to host ");
      ReportCopyToHost(device, sizeof(T) * sz, err);
   }

   void copy_to_host() {
      CHECK_CL_ERROR(err = clEnqueueReadBuffer(device->command_queue(), device_data, CL_TRUE, 0, sizeof(T) * num_elements, (void*) (host_data), 0, NULL, NULL), "Copying to host ");
      ReportCopyToHost(device, sizeof(T) * num_elements, err);
   }
   ////////////////////////////////////////////////
   void init_on_device(const T & val) {
      device->init_on_device(device_data, num_elements, val);
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
   DevicePtrType & device_ptr(void) {
      return device_data;
   }
   HostPtrType & host_ptr(void) {
      return host_data;
   }
   Array<T> * get_array_ptr(void) {
      return this;
   }
   DevicePtrType get_sub_array(size_t start, size_t num_items) {
      cl_buffer_region buff;
      buff.origin = start * sizeof(T);
      buff.size = num_items * sizeof(T);
      cl_int err;
      cl_mem res = clCreateSubBuffer(device_data, allocation_flags, CL_BUFFER_CREATE_TYPE_REGION, &buff, &err);
      galois::opencl::CHECK_CL_ERROR(err, "Failed to create sub-region!");
      return res;
   }

protected:
};

} //end namespace opencl
} //end namespace galois

#endif /* GALOISGPU_OCL_ARRAYIMPL_H_ */
