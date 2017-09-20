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
#include <vector>

#ifndef GALOISGPU_OCL_CL_PLATFORM_H_
#define GALOISGPU_OCL_CL_PLATFORM_H_
namespace galois{
namespace opencl{

struct CL_Platform {
   cl_platform_id id;
   std::string vendor_name;
   std::vector<CL_Device*> devices;
   explicit CL_Platform(cl_platform_id _id) :
         vendor_name("") {
      initialize(_id);
   }
   ~CL_Platform() {
      for(int i=0; i<devices.size(); ++i){
         delete devices[i];
      }
      devices.clear();
      fprintf(stderr, "Released platform:: %s \n", vendor_name.c_str());
   }

   void initialize(cl_platform_id _id) {
      id = _id;
      char string_holder[4 * 256];
      clGetPlatformInfo(id, CL_PLATFORM_NAME, 256, string_holder, NULL);
      std::cout << "Platform: " << string_holder;
      clGetPlatformInfo(id, CL_DEVICE_VENDOR, 256, string_holder, NULL);
      std::cout << ", Vendor: " << string_holder;
      vendor_name = string_holder;
      CHECK_CL_ERROR(clGetPlatformInfo(id, CL_PLATFORM_VERSION, 256, string_holder, NULL), "clGetPlatform info, version");
      std::cout << "[Version: [" << string_holder << "] ";
      cl_uint num_devices;
      clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, 0, 0, &num_devices);
      cl_device_id * l_devices = new cl_device_id[num_devices];
      clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, num_devices, l_devices, 0);
      std::cout << ",devices=" << num_devices << " \n";
      for (unsigned int curr_device = 0; curr_device < num_devices; ++curr_device) {
         CL_Device * d = new CL_Device(this, l_devices[curr_device]);
         devices.push_back(d);
         fprintf(stderr, "Device: %s ,id :: %ld\n", d->name().c_str(), (long) d->id());
      } //For each device
      delete[] l_devices;
   }
   void all_devices(std::vector<cl_device_id> & r) const {
      for (auto d : devices) {
         r.push_back(d->id());
      }

   }
   std::string get_cl_compiler_flags() const {
      /*
       *-cl-single-precision-constant
       *-cl-denorms-are-zero
       *-cl-opt-disable
       *-cl-mad-enable
       *-cl-no-signed-zeros
       *-cl-unsafe-math-optimizations
       *-cl-finite-math-only
       *-cl-fast-relaxed-math
       *-cl-std=<CL1.1>
       *
       * */
//      fprintf(stderr, "Compile Platform:: %s \t", vendor_name.c_str());
      if (vendor_name.find("NVIDIA") != std::string::npos) {
         return "-DCL_NVIDIA";
      } else if (vendor_name.find("AMD") != std::string::npos) {
         return "-DCL_INTEL -g -I. -fbin-source -fbin-llvmir -fbin-amdil";
      } else if (vendor_name.find("Intel") != std::string::npos) {
         return "-DCL_INTEL -g -save-temps -x spir"; //-s /h2/rashid/workspace/GaloisGPU/GaloisGPU/apps/sgd/sgd_static_operator.cl
      }
      return "";
   }
};
}//namespace opencl
} // namespace galois


#endif /* GALOISGPU_OCL_CL_PLATFORM_H_ */
