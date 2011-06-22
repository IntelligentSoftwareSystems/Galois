/*
 * Clustering.cpp
 *
 *  Created on: Jun 22, 2011
 *      Author: rashid
 */
#include<iostream>
#include<vector>
#include"LeafNode.h"
#include"NodeWrapper.h"
#include<stdlib.h>
#include"KdTree.h"
void loopBody(NodeWrapper * cluster, KdTree *kdTree,
		std::vector<NodeWrapper*> * wl, std::vector<ClusterNode*> *clusterArr,
		std::vector<float> * floatArr);

std::vector<LeafNode*> randomGenerate(int count) {
	//RandomGenerator ranGen = new RandomGenerator(12);
	srand(12);
	std::vector<LeafNode*> lights;// = new LeafNode[count];
	float dirX = 0;
	float dirY = 0;
	float dirZ = 1;
	AbstractNode::setGlobalMultitime();
	AbstractNode::setGlobalNumReps();
	//generating random lights
	for (int i = 0; i < count; i++) {
		float x = (float) rand() / (rand() + 1);
		float y = (float) rand() / (rand() + 1);
		float z = (float) rand() / (rand() + 1);
		LeafNode ln(x, y, z, dirX, dirY, dirZ);
		//lights.push_back( new LeafNode(x, y, z, dirX, dirY, dirZ));
		lights.push_back(&ln);
	}
	return lights;
}

void clustering(std::vector<LeafNode*> *inLights) {
	//Launcher launcher = Launcher.getLauncher();
	//used to choose which light is the representative light
	//final RandomGenerator repRanGen = new RandomGenerator(4523489623489L);
	srand(452);

	int tempSize = (1 << NodeWrapper::CONE_RECURSE_DEPTH) + 1;
	std::vector<float> floatArr = std::vector<float>(3 * tempSize);
	//temporary arrays used during cluster construction
	std::vector<ClusterNode *> clusterArr = std::vector<ClusterNode*>(tempSize);
	int numLights = inLights->size();

	std::vector<NodeWrapper*> initialWorklist(numLights);// = new NodeWrapper[numLights];
	for (unsigned int i = 0; i < numLights; i++) {
		std::cout<<"Initializing ... " << i << std::endl;
		NodeWrapper * clusterWrapper = new NodeWrapper((*inLights)[i]);
		initialWorklist[i] = clusterWrapper;
	}

	//launcher.startTiming();
	KdTree * kdTree = KdTree::createTree(&initialWorklist);
	while (initialWorklist.size() > 0) {
		NodeWrapper* currWork = initialWorklist.back();
		initialWorklist.pop_back();
		loopBody(currWork, kdTree, &initialWorklist, &clusterArr, &floatArr);
	}
	// O(1) operation, there is no copy of data but just the creation of an arraylist backed by 'initialWorklist'
	//    const  std::vector<NodeWrapper> wrappers = Arrays.asList(initialWorklist);
	/*
	 GaloisRuntime.foreach(wrappers, new Lambda2Void<NodeWrapper, ForeachContext<NodeWrapper>>() {
	 @Override
	 public void call(final NodeWrapper cluster, final ForeachContext<NodeWrapper> ctx) {
	 NodeWrapper current = cluster;
	 while (current != null && kdTree.contains(current)) {
	 NodeWrapper match = kdTree.findBestMatch(current);
	 if (match == null) {
	 break;
	 }
	 final NodeWrapper matchMatch = kdTree.findBestMatch(match);
	 if (current == matchMatch) {
	 if (kdTree.remove(match)) {
	 NodeWrapper newCluster = new NodeWrapper(current, match, floatArr, clusterArr, repRanGen);
	 ctx.add(newCluster);
	 kdTree.add(newCluster);
	 kdTree.remove(current);
	 }
	 break;
	 } else {
	 if (current == cluster) {
	 ctx.add(current);
	 }
	 current = match;
	 }
	 }
	 }
	 }, Priority.first(ChunkedFIFO.class));
	 */
	//launcher.stopTiming();
	/*if (launcher.isFirstRun()) {
	 std::cout<<"verifying... "<<std::endl;
	 NodeWrapper retval = kdTree.getAny(0.5);
	 AbstractNode serialRoot = new SerialMain().clustering(inLights);
	 if (!serialRoot.equals(retval.light)) {
	 std::cout<<"verification failed!"<<std::endl;
	 }
	 std::cout<<"okay."<<std::endl;
	 }*/
}

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
		if (current == matchMatch) {
			if (kdTree->remove(match)) {
				NodeWrapper *newCluster = new NodeWrapper(current, match,
						*floatArr, *clusterArr);
				wl->push_back(newCluster);
				//	              ctx.add(newCluster);
				kdTree->add(newCluster);
				kdTree->remove(current);
			}
			break;
		} else {
			if (current == cluster) {
				//	              ctx.add(current);
				wl->push_back(current);
			}
			current = match;
		}
	}
}
int main(int argc, const char **argv) {
	int numPoints = 1000;
	std::vector<LeafNode*> points = randomGenerate(numPoints);
	std::cout << "Generated random points " << numPoints << std::endl;
	clustering(&points);
	return 0;
}

// vim:sw=2:sts=2:ts=8
