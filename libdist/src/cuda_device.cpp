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
  auto& net = galois::runtime::getSystemNetworkInterface();
  unsigned host_id = net.ID;
  unsigned num_hosts = net.Num;
  unsigned num_hosts_per_node = num_hosts / num_nodes;
  assert((num_hosts % num_nodes) == 0);
  assert(personality_set.length() == num_hosts_per_node);
  unsigned num_gpus_per_node = std::count(personality_set.begin(), personality_set.end(), 'g');
  unsigned num_gpus_before = std::count(personality_set.begin(), 
                                        personality_set.begin() + 
                                        (host_id % num_hosts_per_node), 'g');
  return (num_gpus_before % num_gpus_per_node);
}
