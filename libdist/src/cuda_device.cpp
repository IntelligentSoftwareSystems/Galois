/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

/*
 */

/**
 * @file cuda_device.cpp
 *
 * Contains implementation for function that gets gpu device ID.
 */
#include "galois/cuda/HostDecls.h"
#include "galois/Galois.h"
#include "galois/runtime/Network.h"
#include <algorithm>

int get_gpu_device_id(std::string personality_set, int num_nodes) {
  auto& net                   = galois::runtime::getSystemNetworkInterface();
  unsigned host_id            = net.ID;
  unsigned num_hosts          = net.Num;
  unsigned num_hosts_per_node = num_hosts / num_nodes;
  assert((num_hosts % num_nodes) == 0);
  assert(personality_set.length() == num_hosts_per_node);
  unsigned num_gpus_per_node =
      std::count(personality_set.begin(), personality_set.end(), 'g');
  unsigned num_gpus_before =
      std::count(personality_set.begin(),
                 personality_set.begin() + (host_id % num_hosts_per_node), 'g');
  return (num_gpus_before % num_gpus_per_node);
}
