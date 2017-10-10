/**  Agglomerative Clustering -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * @author Rashid Kaleem <rashid.kaleem@gmail.com>
 */
#ifndef POTENTIALCLUSTER_H_
#define POTENTIALCLUSTER_H_
#include "NodeWrapper.h"
#include <limits>

using namespace std;

class PotentialCluster {
public:
  const NodeWrapper& original;
  NodeWrapper* closest;
  double clusterSize;

  PotentialCluster(NodeWrapper& pOriginal) : original(pOriginal) {
    closest     = NULL;
    clusterSize = numeric_limits<float>::max();
  }
  friend ostream& operator<<(ostream& s, const PotentialCluster& p);
};

ostream& operator<<(ostream& s, const PotentialCluster& p) {
  s << "PC : [" << p.original << ", ?";
  if (p.closest != NULL)
    s << *(p.closest);
  else
    s << "NULL";
  s << "," << p.clusterSize << "]";
  return s;
}

#endif /* POTENTIALCLUSTER_H_ */
