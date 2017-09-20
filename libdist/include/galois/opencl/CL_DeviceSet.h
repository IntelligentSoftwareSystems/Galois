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
#include "CL_Header.h"
#ifndef GALOISGPU_OCL_DEVICESET_H_
#define GALOISGPU_OCL_DEVICESET_H_

namespace galois{
namespace opencl{
struct DeviceSet {
   std::vector<std::string> names;
   std::vector<CL_Device *> devices;
   std::vector<float> ratios;
   DeviceSet() {
      ///TODO
   }
   DeviceSet(std::string filename) {
      init(filename);
   }
   void reset() {
      names.clear();
      devices.clear();
      ratios.clear();
   }
   size_t num_devices() const {
      return devices.size();
   }
   void init(std::string filename) {
      std::ifstream file(filename);
      ratios.push_back(0.0);
      while (file.eof() == false) {
         int platform, device;
         float workload;
         std::string name;
         file >> platform;
         file >> device;
         file >> workload;
         file >> name;
         //must have non-empty name to help avoid portability issues.
         if (workload > 0 && name.size() > 0) {
            devices.push_back(getCLContext()->get_device(platform, device));
            names.push_back(name);
//            fprintf(stderr, "Loaded [%d, %d, %6.6g, %s]=>%s\n",platform, device, workload, name.c_str(),devices[devices.size()-1]->name().c_str() );
            ratios.push_back(workload);
         }
      }
      for (size_t i = 1; i < ratios.size(); ++i) {
         ratios[i] += ratios[i - 1];
      }
      for (size_t i = 0; i < devices.size(); ++i) {
         fprintf(stderr, "%s -> %s -> %6.6g \n", names[i].c_str(), devices[i]->name().c_str(), ratios[i + 1]);
      }
      assert(ratios[ratios.size() - 1] == 1.0 && "Work distribution between devices must sum to 1.0!");
   }
};
}//namespace opencl
} // namespace galois




#endif /* GALOISGPU_OCL_DEVICESET_H_ */
