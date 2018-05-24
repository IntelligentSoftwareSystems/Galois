#ifndef GALOISGPU_OCL_DEVICESTATS_H_
#define GALOISGPU_OCL_DEVICESTATS_H_


namespace galois {
namespace opencl {
struct DeviceStats {
   long copied_to_device;
   long copied_to_host;
   long max_allocated;
   long allocated;
   DeviceStats() :
         copied_to_device(0), copied_to_host(0), max_allocated(0), allocated(0) {
   }
   DeviceStats & operator=(const DeviceStats & other) {
      this->copied_to_device = other.copied_to_device;
      this->copied_to_host = other.copied_to_host;
      this->max_allocated = other.max_allocated;
      this->allocated = other.allocated;
      return *this;
   }
   DeviceStats operator-(const DeviceStats & other) const {
      DeviceStats res;
      res.copied_to_device = this->copied_to_device - other.copied_to_device;
      res.copied_to_host = this->copied_to_host - other.copied_to_host;
      res.allocated = this->allocated - other.allocated;
      res.max_allocated = 0; //this->max_allocated = other.max_allocated;
      return res;
   }
   static float toMB(long v) {
      return v / (float) (1024 * 1024);
   }
   void print() {
      fprintf(stderr, "Allocated, %6.6g, MaxAllocated, %6.6g, CopiedToHost, %6.6g, CopiedToDevice, %6.6g", toMB(allocated), toMB(max_allocated), toMB(copied_to_host),
            toMB(copied_to_device));
   }
   void print_long() {
         fprintf(stderr, "Allocated, %6.6g \nMaxAllocated, %6.6g \nCopiedToHost, %6.6g \nCopiedToDevice, %6.6g", toMB(allocated), toMB(max_allocated), toMB(copied_to_host),
               toMB(copied_to_device));
      }
};
}//namespace opencl
}//namespace galois

#endif /* GALOISGPU_OCL_DEVICESTATS_H_ */
