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
#include"KdTree.h"
#include"ClusteringSerial.h"
#include<stdlib.h>
static const char* name = "Agglomorative Clustering";
static const char* description =
"Computes the Minimal Spanning Tree using Boruvka\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/agglomerativeclustering.html";
static const char* help = "[num points]";



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
struct FindMatching {
	KdTree * kdTree;
	//std::set<NodeWrapper*> *wrappers;
	GaloisRuntime::galois_insert_bag<NodeWrapper *> &newWl;
	std::map<NodeWrapper*, NodeWrapper*> &matchings;
	//FindMatching(KdTree *& pk, std::set<NodeWrapper*> *& pW, std::set<NodeWrapper*> & pNW, std::map<NodeWrapper*, NodeWrapper*> &pM):
	FindMatching(KdTree *& pk, GaloisRuntime::galois_insert_bag<NodeWrapper *> & pNW, std::map<NodeWrapper*, NodeWrapper*> &pM):
		kdTree(pk), newWl(pNW), matchings(pM)
	{
	}	template<typename ContextTy>
	void __attribute__((noinline)) operator()(NodeWrapper * cluster, ContextTy& lwl) {
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
					newWl.push(current);
				}
				current = match;
			}
		}

	}
};
//*********************************************************************
struct PerformMatching {
	KdTree *& kdTree;
	//std::set<NodeWrapper*> *&wrappers;
//	std::vector<NodeWrapper *> &newWl;
	GaloisRuntime::galois_insert_bag<NodeWrapper *> &newWl;
	std::map<NodeWrapper*, NodeWrapper*> &matchings;
	std::vector<float>  &floatArr;
	std::vector<ClusterNode *> &clusterArr;
	//PerformMatching(KdTree *& pk, std::set<NodeWrapper*> *& pW, std::set<NodeWrapper*> & pNW, std::map<NodeWrapper*, NodeWrapper*> &pM, std::vector<float> & pF, std::vector<ClusterNode*>& pC):
	PerformMatching(KdTree *& pk,GaloisRuntime::galois_insert_bag<NodeWrapper *> & pNW, std::map<NodeWrapper*, NodeWrapper*> &pM, std::vector<float> & pF, std::vector<ClusterNode*>& pC):
		kdTree(pk), newWl(pNW), matchings(pM), floatArr(pF), clusterArr(pC)
	{
	} 
	template<typename ContextTy>
		void __attribute__((noinline)) operator()(pair<NodeWrapper *,NodeWrapper*> &itM, ContextTy& lwl) {
			NodeWrapper *match = (itM).second;
			NodeWrapper *current = (itM).first;
			if (matchings.count((itM).second))
				if (match < current)
					return;
			if (kdTree->remove(match)) {
				NodeWrapper *newCluster = new NodeWrapper(current, (match), floatArr, clusterArr);
				newWl.push(newCluster);
				kdTree->add(newCluster);
				kdTree->remove(current);
			}
		}
};

//*********************************************************************
void runGaloisBody (std::vector<LeafNode*>* inLights) {
	int tempSize = (1 << NodeWrapper::CONE_RECURSE_DEPTH) + 1;
	std::vector<float> floatArr(3 * tempSize);
	std::vector<ClusterNode *> clusterArr(tempSize);
	int numLights = inLights->size();
	std::vector<LeafNode*> *copyLights = new std::vector<LeafNode*>(numLights);
	std::vector<NodeWrapper*> initialWorklist(numLights);
	using namespace GaloisRuntime::WorkList;
	std::vector<NodeWrapper*> *wrappers = new std::vector<NodeWrapper*>();

	for (int i = 0; i < numLights; i++) {
//		std::cout<<"Serial "<<i<<" :: "<<(*(*inLights)[i])<<std::endl;
		LeafNode * cp = new LeafNode((*inLights)[i]->getX(),(*inLights)[i]->getY(),(*inLights)[i]->getZ(), (*inLights)[i]->getDirX(), (*inLights)[i]->getDirY(), (*inLights)[i]->getDirZ());
		(*copyLights)[i]=cp;
		NodeWrapper *clusterWrapper = new NodeWrapper(*(*inLights)[i]);
		initialWorklist[i] = clusterWrapper;
		wrappers->push_back(clusterWrapper);
	}
	Galois::StatTimer T;
	KdTree *kdTree = KdTree::createTree(&initialWorklist);
//	std::cout<<"Before ::" <<*kdTree<<std::endl;
	T.start();
	//TODO replace w/ vector and list
	GaloisRuntime::galois_insert_bag<NodeWrapper *> *newWl = new GaloisRuntime::galois_insert_bag<NodeWrapper *>();
	std::map<NodeWrapper*, NodeWrapper*> matchings;
	while (wrappers->size() > 1) {
		std::cout<<"Beginning iteration "<<wrappers->size()<<std::endl;
		matchings.clear();
		FindMatching f(kdTree, *newWl,matchings);
		Galois::for_each(wrappers->begin(),wrappers->end(), f);
		std::cout<<"Found matching"<<matchings.size()<<std::endl;
		PerformMatching p(kdTree, *newWl,matchings, floatArr,clusterArr);
		std::vector<pair<NodeWrapper*,NodeWrapper*> > work;
		for(std::map<NodeWrapper*,NodeWrapper*>::iterator it = matchings.begin(), itEnd = matchings.end(); it!=itEnd;it++)
			work.push_back(pair<NodeWrapper*,NodeWrapper*>((*it).first,(*it).second));
		Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<32> >(work.begin(),work.end(), p);
		std::cout<<"Performed matching"<<std::endl;//newWl.size()<<std::endl;
		wrappers->clear();
		for(GaloisRuntime::galois_insert_bag<NodeWrapper *>::iterator itW = newWl->begin(), itWEnd = newWl->end();
				itW!=itWEnd; itW++){
			wrappers->push_back(*itW);
		}
		delete newWl;
		newWl = new GaloisRuntime::galois_insert_bag<NodeWrapper *>();
//		newWl.clear();
	}
	T.stop();
//	std::cout<<"Solution is ::" <<*kdTree<<std::endl;
	{
//		std::cout<<"verifying... "<<std::endl;
//		NodeWrapper * retval = kdTree->getAny(0.5);
//		AbstractNode * serialVal = serialClustering(copyLights);
//		if(serialVal->equals(*retval->light)==false)
//			std::cout<<"Invalid output!";
	}

}
//

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
	//serialClustering(&points);
	runGaloisBody(&points);
	std::cout << "Terminated normally" << std::endl;
	return 0;
}
// vim:sw=2:sts=2:ts=8
