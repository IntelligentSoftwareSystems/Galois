/*
 * PotentialCluster.h
 *
 *  Created on: Jun 22, 2011
 *      Author: rashid
 */
#include"NodeWrapper.h"
#include<limits>
#ifndef POTENTIALCLUSTER_H_
#define POTENTIALCLUSTER_H_
class PotentialCluster{
public:
	  NodeWrapper *original;
	  NodeWrapper  *closest;
	  float clusterSize;

	  PotentialCluster(NodeWrapper *&original) {
	    this->original = original;
	    closest = NULL;
	    clusterSize = std::numeric_limits<float>::max();
	  }
};

#endif /* POTENTIALCLUSTER_H_ */
