/*
 * Clustering.cpp
 *
 *  Created on: Jun 22, 2011
 *      Author: rashid
 */
#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"

#include "Galois/Graphs/Serialize.h"
#include "Galois/Graphs/FileGraph.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"
#include<vector>
#include<map>
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <set>

#include"LeafNode.h"
#include"NodeWrapper.h"
#include<stdlib.h>
#include"KdTree.h"
static const char* name = "Agglomorative Clustering";
static const char* description =
"Computes the Minimal Spanning Tree using Boruvka\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/agglomerativeclustering.html";
static const char* help = "[num points]";


void loopBody(NodeWrapper * cluster, KdTree *kdTree, std::vector<NodeWrapper*> * wl, std::vector<ClusterNode*> *clusterArr, std::vector<
		float> * floatArr);
//*********************************************************************
std::vector<LeafNode*> randomGenerate(int count) {
	std::vector<LeafNode*> lights;
	float dirX = 0;
	float dirY = 0;
	float dirZ = 1;
	//	std::cout << "Setting global multi time to false" << std::endl;
	AbstractNode::setGlobalMultitime();
	//	std::cout << "Done\nSetting global num reps." << std::endl;
	AbstractNode::setGlobalNumReps();
	//	std::cout << "About to create lights" << std::endl;
	//generating random lights
	for (int i = 0; i < count; i++) {
		float x = ((float) rand()) / std::numeric_limits<int>::max();
		float y = ((float) rand()) / std::numeric_limits<int>::max();
		float z = ((float) rand()) / std::numeric_limits<int>::max();
		LeafNode * ln = new LeafNode(x, y, z, dirX, dirY, dirZ);
		lights.push_back(ln);
	}
	//	std::cout << "Done creating lights" << std::endl;
	return lights;
}
//*********************************************************************
void clustering(std::vector<LeafNode*> *inLights) {
	//used to choose which light is the representative light

	int tempSize = (1 << NodeWrapper::CONE_RECURSE_DEPTH) + 1;
	std::vector<float> floatArr(3 * tempSize);
	std::vector<ClusterNode *> clusterArr(tempSize);
	int numLights = inLights->size();
	std::vector<NodeWrapper*> initialWorklist(numLights);
	for (int i = 0; i < numLights; i++) {
		NodeWrapper * clusterWrapper = new NodeWrapper(*(*inLights)[i]);
		initialWorklist[i] = clusterWrapper;
	}
	KdTree * kdTree = KdTree::createTree(&initialWorklist);
	while (initialWorklist.size() > 0) {
		NodeWrapper* currWork = initialWorklist.back();
		initialWorklist.pop_back();
		loopBody(currWork, kdTree, &initialWorklist, &clusterArr, &floatArr);
	}
	NodeWrapper *retval = kdTree->getAny(0.5);
}
//*********************************************************************
struct findMatching {
	KdTree * kdTree;
	std::set<NodeWrapper*> *wrappers;
	std::set<NodeWrapper *> newWl;
		std::map<NodeWrapper*, NodeWrapper*> matchings;
	template<typename ContextTy>
		void __attribute__((noinline)) operator()(NodeWrapper * curr, ContextTy& lwl) {
		}
};
//*********************************************************************
struct performMatching {
	template<typename ContextTy>
		void __attribute__((noinline)) operator()(pair<NodeWrapper *,NodeWrapper*> match, ContextTy& lwl) {
		}
};

//*********************************************************************
void loopBody(NodeWrapper * cluster, KdTree *kdTree, std::vector<NodeWrapper*> * wl, std::vector<ClusterNode*> *clusterArr, std::vector<
		float> * floatArr) {
	NodeWrapper *current = cluster;
	while (current != NULL && kdTree->contains(current)) {
		NodeWrapper *match = kdTree->findBestMatch(current);
		if (match == NULL) {
			break;
		}
		NodeWrapper *matchMatch = kdTree->findBestMatch(match);
		if (current->equals(matchMatch)) {
			if (kdTree->remove(match)) {
				NodeWrapper *newCluster = new NodeWrapper(current, (match), *floatArr, *clusterArr);
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
void findMatchings(KdTree *& kdTree, std::set<NodeWrapper*> *&wrappers,std::set<NodeWrapper *> &newWl,
		std::map<NodeWrapper*, NodeWrapper*> &matchings){
	for (std::set<NodeWrapper*>::iterator it = wrappers->begin(), itEnd = wrappers->end(); it != itEnd; it++) {
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
void performMatchings(KdTree *& kdTree, std::set<NodeWrapper*> *&wrappers,std::set<NodeWrapper *> &newWl,
		std::map<NodeWrapper*, NodeWrapper*> &matchings, std::vector<float>  &floatArr,  std::vector<ClusterNode *> &clusterArr){
	for (std::map<NodeWrapper*, NodeWrapper*>::iterator itM = matchings.begin(), itMEnd = matchings.end(); itM != itMEnd; itM++) {

		NodeWrapper *match = (*itM).second;
		NodeWrapper *current = (*itM).first;
		if (matchings.count((*itM).second))
			if (match < current)
				continue;
		if (kdTree->remove(match)) {
			NodeWrapper *newCluster = new NodeWrapper(current, (match), floatArr, clusterArr);
			newWl.insert(newCluster);
			kdTree->add(newCluster);
			kdTree->remove(current);
		}

	}
	wrappers->clear();
	for(std::set<NodeWrapper*>::iterator itW = newWl.begin(), itWEnd = newWl.end();
			itW!=itWEnd; itW++){
		wrappers->insert(*itW);
	}
	newWl.clear();
}
//******************************************************************************************

void serialClustering(std::vector<LeafNode*>* inLights) {
	int tempSize = (1 << NodeWrapper::CONE_RECURSE_DEPTH) + 1;
	std::vector<float> floatArr(3 * tempSize);
	std::vector<ClusterNode *> clusterArr(tempSize);
	int numLights = inLights->size();
	std::vector<NodeWrapper*> initialWorklist(numLights);
	std::set<NodeWrapper*> *wrappers = new std::set<NodeWrapper*>();
	for (int i = 0; i < numLights; i++) {
		NodeWrapper *clusterWrapper = new NodeWrapper(*(*inLights)[i]);
		initialWorklist[i] = clusterWrapper;
		wrappers->insert(clusterWrapper);
	}
	//	launcher.startTiming();
	//Galois::StatTime T;
	//T.start();
	KdTree *kdTree = KdTree::createTree(&initialWorklist);
	// O(1) operation, there is no copy of data but just the creation of an
	// arraylist backed by 'initialWorklist'
	std::set<NodeWrapper *> newWl;
	std::map<NodeWrapper*, NodeWrapper*> matchings;
	while (wrappers->size() > 1) {
		matchings.clear();
		findMatchings(kdTree, wrappers,newWl,matchings);
		performMatchings(kdTree,wrappers, newWl,matchings, floatArr,clusterArr);
	}
	//T.stop();
	{
		std::cout<<"verifying... "<<std::endl;
		NodeWrapper * retval = kdTree->getAny(0.5);
	}

}
//******************************************************************************************

int main(int argc, const char **argv) {
	std::vector<const char*> args = parse_command_line(argc, argv, help);
	int numPoints=0;  
	if (args.size() == 1) {
		numPoints = atoi(args[0]);
	}
	else if(args.size()==0){
		numPoints = 1000;
	}
	else{
		std::cerr<<"Invalid number of args"<<std::endl;
		return -1;
	}

	printBanner(std::cout, name, description, url);

	std::vector<LeafNode*> points = randomGenerate(numPoints);
	std::cout << "Generated random points " << numPoints << std::endl;
	serialClustering(&points);
	std::cout << "Terminated normally" << std::endl;
	return 0;
}
// vim:sw=2:sts=2:ts=8
