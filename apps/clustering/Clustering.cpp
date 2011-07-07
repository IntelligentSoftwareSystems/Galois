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
#include <sys/time.h>
static const char* name = "Unordered Agglomerative Clustering";
static const char* description = "Unordered Implementation of the well-known data-mining algorithm\n";
static const char* url = "http://iss.ices.utexas.edu/lonestar/agglomerativeclustering.html";
static const char* help = "[num points]";



//*********************************************************************
std::vector<LeafNode*> randomGenerate(int count) {
	std::vector<LeafNode*> lights;
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
		LeafNode * ln = new LeafNode(x, y, z, 0, 0, 1);
		lights.push_back(ln);
	}
	//	std::cout << "Done creating lights" << std::endl;
	return lights;
}

//*********************************************************************
struct FindMatching {
	KdTree * kdTree;
	GaloisRuntime::galois_insert_bag<NodeWrapper *> *&newWl;
	GaloisRuntime::galois_insert_bag<std::pair<NodeWrapper *,NodeWrapper*> > * &matchings;
	FindMatching(KdTree *& pk, GaloisRuntime::galois_insert_bag<NodeWrapper *> *& pNW, GaloisRuntime::galois_insert_bag<std::pair<NodeWrapper *,NodeWrapper*> >* &pM):
		kdTree(pk), newWl(pNW), matchings(pM)
	{
	}	template<typename ContextTy>
	void __attribute__((noinline)) operator()(NodeWrapper *& cluster, ContextTy& lwl) {
		NodeWrapper *current = cluster;
		while (current != NULL && kdTree->contains(current)) {
			NodeWrapper *match = kdTree->findBestMatch(current);
			if (match == NULL) {
				break;
			}
			NodeWrapper *matchMatch = kdTree->findBestMatch(match);
			if (current->equals(matchMatch)) {
				//Use a tie breaking policy. Since a->b and b->a are both
				//matchings, we only insert one. This makes it easier for the
				//second process body to do its work.
				if(current<match)
					matchings->push(std::pair<NodeWrapper*,NodeWrapper*>(current,match));
				break;
			} else {
				if (current == cluster) {
					newWl->push(current);
				}
				current = match;
			}
		}

	}
};
//*********************************************************************
struct PerformMatching {
	KdTree *& kdTree;
	GaloisRuntime::galois_insert_bag<NodeWrapper *>* &newWl;
	GaloisRuntime::galois_insert_bag<pair<NodeWrapper *,NodeWrapper*> > *&matchings;
	std::vector<float>  &floatArr;
	std::vector<ClusterNode *> &clusterArr;
//	Galois::Allocator<NodeWrapper> alloc;
		PerformMatching(KdTree *& pk,GaloisRuntime::galois_insert_bag<NodeWrapper *> *& pNW, GaloisRuntime::galois_insert_bag<std::pair<NodeWrapper *,NodeWrapper*> > * &pM, std::vector<float> & pF, std::vector<ClusterNode*>& pC):
		kdTree(pk), newWl(pNW), matchings(pM), floatArr(pF), clusterArr(pC)
	{
//			Galois::Allocator<NodeWrapper> alloc;
	} 
	template<typename ContextTy>
		void __attribute__((noinline)) operator()(pair<NodeWrapper *,NodeWrapper*> &itM, ContextTy& lwl) {
			NodeWrapper *match = (itM).second;
			NodeWrapper *current = (itM).first;
			if (kdTree->remove(match)) {
				NodeWrapper *newCluster = new NodeWrapper(current, (match), floatArr, clusterArr);
//				alloc.allocate(sizeof(NodeWrapper));
//				NodeWrapper *newCluster = alloc.allocate(sizeof(NodeWrapper)); NodeWrapper(current, (match), floatArr, clusterArr);
				newWl->push(newCluster);
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
	std::vector<NodeWrapper*> *initialWorklist= new std::vector<NodeWrapper *>(numLights);
	using namespace GaloisRuntime::WorkList;
	std::vector<NodeWrapper*> *wrappers = new std::vector<NodeWrapper*>();
	for (int i = 0; i < numLights; i++) {
//		std::cout<<"Serial "<<i<<" :: "<<(*(*inLights)[i])<<std::endl;
		NodeWrapper *clusterWrapper = new NodeWrapper(*(*inLights)[i]);
		(*initialWorklist)[i] = clusterWrapper;
		wrappers->push_back(clusterWrapper);
	}
	Galois::StatTimer T;
	KdTree *kdTree = &KdTree::createTree(initialWorklist);
//	std::cout<<"Before ::" <<*kdTree<<std::endl;
	//TODO replace w/ vector and list
	GaloisRuntime::galois_insert_bag<NodeWrapper *> *newWl = new GaloisRuntime::galois_insert_bag<NodeWrapper *>();
	GaloisRuntime::galois_insert_bag<pair<NodeWrapper *,NodeWrapper*> > * matchings = new GaloisRuntime::galois_insert_bag<std::pair<NodeWrapper *,NodeWrapper*> >();
	FindMatching f(kdTree, newWl,matchings);
	PerformMatching p(kdTree, newWl,matchings, floatArr,clusterArr);
	T.start();
	timeval start,stop,result;
	gettimeofday(&start,NULL);

	while (wrappers->size() > 1) {
//		std::cout<<"Beginning iteration "<<wrappers->size()<<std::endl;
		matchings = new GaloisRuntime::galois_insert_bag<std::pair<NodeWrapper *,NodeWrapper*> >();
		f.matchings=matchings;
		f.newWl=newWl;
		p.newWl = newWl;
		p.matchings=matchings;
		Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<32> >(wrappers->begin(),wrappers->end(), f);
//		std::cout<<"Found matching"<<matchings.size()<<std::endl;
		std::vector<pair<NodeWrapper*,NodeWrapper*> > work;
		for(GaloisRuntime::galois_insert_bag<std::pair<NodeWrapper *,NodeWrapper*> >::iterator it = matchings->begin(), itEnd = matchings->end(); it!=itEnd;it++)
			work.push_back(pair<NodeWrapper*,NodeWrapper*>((*it).first,(*it).second));
		Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<32> >(work.begin(),work.end(), p);
//		std::cout<<"Performed matching"<<std::endl;//newWl.size()<<std::endl;
		wrappers->clear();
		for(GaloisRuntime::galois_insert_bag<NodeWrapper *>::iterator itW = newWl->begin(), itWEnd = newWl->end();
				itW!=itWEnd; itW++){
			wrappers->push_back(*itW);
		}
		delete newWl;
		delete matchings;
		newWl = new GaloisRuntime::galois_insert_bag<NodeWrapper *>();
	}
	gettimeofday(&stop,NULL);
	timersub(&stop,&start,&result);
	double t = result.tv_sec + result.tv_usec/1000000.0; // 1000000 microseconds per second
	std::cout<<"Time inside loop :: "<<t<<std::endl;
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
//	std::cout << "Generated random points " << numPoints << std::endl;
	runGaloisBody(&points);
	std::cout << "Terminated normally" << std::endl;
	return 0;
}
// vim:sw=2:sts=2:ts=8
