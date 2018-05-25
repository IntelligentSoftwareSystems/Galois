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

void internal::heteroSetup(std::vector<unsigned>& scaleFactor) {
  const unsigned my_host_id = galois::runtime::getHostID();

  // Parse arg string when running on multiple hosts and update/override 
  // personality with corresponding value.
  auto& net = galois::runtime::getSystemNetworkInterface();

  if (num_nodes == -1) num_nodes = net.Num;

  assert((net.Num % num_nodes) == 0);

  if (personality_set.length() == (net.Num / num_nodes)) {
    switch (personality_set.c_str()[my_host_id % (net.Num / num_nodes)]) {
      case 'g':
        personality = GPU_CUDA;
        break;
      case 'o':
        assert(0);
        personality = GPU_OPENCL;
        break;
      case 'c':
      default:
        personality = CPU;
        break;
    }

    if (personality == GPU_CUDA) {
      gpudevice = get_gpu_device_id(personality_set, num_nodes);
    } else {
      gpudevice = -1;
    }

    // scale factor setup
    if ((scalecpu > 1) || (scalegpu > 1)) {
      for (unsigned i = 0; i < net.Num; ++i) {
        if (personality_set.c_str()[i % num_nodes] == 'c') {
          scaleFactor.push_back(scalecpu);
        } else {
          scaleFactor.push_back(scalegpu);
        }
      }
    }
  }
}
#endif
