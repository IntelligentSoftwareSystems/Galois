#ifndef POTENTIALCLUSTER_H_
#define POTENTIALCLUSTER_H_
#include "NodeWrapper.h"
#include <limits>

class PotentialCluster {
public:
  const NodeWrapper& original;
  NodeWrapper* closest;
  double clusterSize;

  PotentialCluster(NodeWrapper& pOriginal) : original(pOriginal) {
    closest     = NULL;
    clusterSize = std::numeric_limits<float>::max();
  }
  friend std::ostream& operator<<(std::ostream& s, const PotentialCluster& p);
};

std::ostream& operator<<(std::ostream& s, const PotentialCluster& p) {
  s << "PC : [" << p.original << ", ?";
  if (p.closest != NULL)
    s << *(p.closest);
  else
    s << "NULL";
  s << "," << p.clusterSize << "]";
  return s;
}

#endif /* POTENTIALCLUSTER_H_ */
