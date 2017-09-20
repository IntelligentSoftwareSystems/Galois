#include "Galois/Runtime/Cuda/cuda_device.h"
#include "Galois/Galois.h"
#include "Galois/Runtime/Network.h"
#include <algorithm>

int get_gpu_device_id(std::string personality_set, int num_nodes){
  auto& net = galois::Runtime::getSystemNetworkInterface();
  unsigned host_id = net.ID;
  unsigned num_hosts = net.Num;
  unsigned num_hosts_per_node = num_hosts / num_nodes;
  assert((num_hosts % num_nodes) == 0);
  assert(personality_set.length() == num_hosts_per_node);
  unsigned num_gpus_per_node = std::count(personality_set.begin(), personality_set.end(), 'g');
  unsigned num_gpus_before = std::count(personality_set.begin(), personality_set.begin() + (host_id % num_hosts_per_node), 'g');
  return (num_gpus_before % num_gpus_per_node);
}

