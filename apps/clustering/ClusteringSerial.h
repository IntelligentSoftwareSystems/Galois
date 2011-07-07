/** Unordered Agglomerative Clustering -*- C++ -*-
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
 * @author Rashid Kaleem <rashid@cs.utexas.edu>
 */

#ifndef CLUSTERING_SERIAL_H
#define CLUSTERING_SERIAL_H
//*********************************************************************
void loopBody(NodeWrapper * cluster, KdTree *kdTree, std::vector<NodeWrapper*> * wl, std::vector<ClusterNode*> *clusterArr, std::vector<
		float> * floatArr);
void clustering(std::vector<LeafNode*> *inLights) {
	//used to choose which light is the representative light

	int tempSize = (1 << NodeWrapper::CONE_RECURSE_DEPTH) + 1;
	std::vector<float> floatArr(3 * tempSize);
	std::vector<ClusterNode *> clusterArr(tempSize);
	int numLights = inLights->size();
	std::vector<NodeWrapper*> *initialWorklist= new std::vector<NodeWrapper *>(numLights);
	for (int i = 0; i < numLights; i++) {
		NodeWrapper * clusterWrapper = new NodeWrapper(*(*inLights)[i]);
		(*initialWorklist)[i] = clusterWrapper;
	}
	KdTree * kdTree = &KdTree::createTree(initialWorklist);
	while (initialWorklist->size() > 0) {
		NodeWrapper* currWork = initialWorklist->back();
		initialWorklist->pop_back();
		loopBody(currWork, kdTree, initialWorklist, &clusterArr, &floatArr);
	}
}
//*********************************************************************
void loopBody(NodeWrapper * cluster, KdTree *kdTree,
		std::vector<NodeWrapper*> * wl, std::vector<ClusterNode*> *clusterArr,
		std::vector<float> * floatArr) {
	NodeWrapper *current = cluster;
	while (current != NULL && kdTree->contains(current)) {
		NodeWrapper *match = kdTree->findBestMatch(current);
		if (match == NULL) {
			break;
		}
		NodeWrapper *matchMatch = kdTree->findBestMatch(match);
		if (current->equals(matchMatch)) {
			if (kdTree->remove(match)) {
				NodeWrapper *newCluster = new NodeWrapper(current, (match),
						*floatArr, *clusterArr);
				wl->push_back(newCluster);
				kdTree->add(newCluster);
				kdTree->remove(current);
			}
			break;
		} else {
			if (current->equals(cluster)) {
				wl->push_back(current);
			}
			current = match;
		}
	}
}
//******************************************************************************************
void findMatchings(KdTree *& kdTree, std::set<NodeWrapper*> *&wrappers,
		std::set<NodeWrapper *> &newWl,
		std::map<NodeWrapper*, NodeWrapper*> &matchings) {
	for (std::set<NodeWrapper*>::iterator it = wrappers->begin(), itEnd =
			wrappers->end(); it != itEnd; it++) {
		NodeWrapper * cluster = *it;
		NodeWrapper *current = cluster;
		while (current != NULL && kdTree->contains(current)) {
			NodeWrapper *match = kdTree->findBestMatch(current);
			if (match == NULL) {
				break;
			}
			NodeWrapper *matchMatch = kdTree->findBestMatch(match);
			if (current->equals(matchMatch)) {
				matchings[current] = match;
				break;
			} else {
				if (current == cluster) {
					newWl.insert(current);
				}
				current = match;
			}
		}

	}
}
//******************************************************************************************
void performMatchings(KdTree *& kdTree, std::set<NodeWrapper*> *&wrappers,
		std::set<NodeWrapper *> &newWl,
		std::map<NodeWrapper*, NodeWrapper*> &matchings,
		std::vector<float> &floatArr, std::vector<ClusterNode *> &clusterArr) {
	for (std::map<NodeWrapper*, NodeWrapper*>::iterator
			itM = matchings.begin(), itMEnd = matchings.end(); itM != itMEnd; itM++) {

		NodeWrapper *match = (*itM).second;
		NodeWrapper *current = (*itM).first;
		if (matchings.count((*itM).second))
			if (match < current)
				continue;
		if (kdTree->remove(match)) {
			NodeWrapper *newCluster = new NodeWrapper(current, (match),
					floatArr, clusterArr);
			newWl.insert(newCluster);
			kdTree->add(newCluster);
			kdTree->remove(current);
		}

	}
	wrappers->clear();
	for (std::set<NodeWrapper*>::iterator itW = newWl.begin(), itWEnd =
			newWl.end(); itW != itWEnd; itW++) {
		wrappers->insert(*itW);
	}
	newWl.clear();
}
//******************************************************************************************

AbstractNode* serialClustering(std::vector<LeafNode*>* inLights) {
	int tempSize = (1 << NodeWrapper::CONE_RECURSE_DEPTH) + 1;
	std::vector<float> floatArr(3 * tempSize);
	std::vector<ClusterNode *> clusterArr(tempSize);
	int numLights = inLights->size();
	std::vector<NodeWrapper*> *initialWorklist= new std::vector<NodeWrapper *>(numLights);
	std::set<NodeWrapper*> *wrappers = new std::set<NodeWrapper*>();
	for (int i = 0; i < numLights; i++) {
		std::cout<<"Serial "<<i<<" :: "<<(*(*inLights)[i])<<std::endl;
		NodeWrapper *clusterWrapper = new NodeWrapper(*(*inLights)[i]);
		(*initialWorklist)[i] = clusterWrapper;
		wrappers->insert(clusterWrapper);
	}
	//	launcher.startTiming();
	//Galois::StatTime T;
	//T.start();
	KdTree *kdTree = &KdTree::createTree(initialWorklist);
	// O(1) operation, there is no copy of data but just the creation of an
	// arraylist backed by 'initialWorklist'
	std::set<NodeWrapper *> newWl;
	std::map<NodeWrapper*, NodeWrapper*> matchings;
	while (wrappers->size() > 1) {
		matchings.clear();
		findMatchings(kdTree, wrappers, newWl, matchings);
		performMatchings(kdTree, wrappers, newWl, matchings, floatArr,
				clusterArr);
	}
	NodeWrapper * retVal = kdTree->getAny(0.5);
	return retVal->light;
}
#endif //CLUSTERING_SERIAL_H
