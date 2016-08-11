#include "Galois/Runtime/Cuda/cuda_device.h"
#include "Galois/Galois.h"
#include <algorithm>

int get_gpu_device_id(std::string personality_set, int num_nodes){
  auto& net = Galois::Runtime::getSystemNetworkInterface();
  unsigned host_id = net.ID;
  unsigned num_hosts = net.Num;
  assert(personality_set.length() == num_hosts);
  if (num_nodes == -1) num_nodes = num_hosts;
  unsigned num_gpus = std::count(personality_set.begin(), personality_set.end(), 'g');
  if ((num_gpus % num_nodes) != 0) return -1;
  unsigned num_gpus_per_node = num_gpus/num_nodes;
  unsigned num_gpus_before = std::count(personality_set.begin(), personality_set.begin() + host_id, 'g');
  return (num_gpus_before % num_gpus_per_node);
}

