/** Single source shortest paths -*- C++ -*-
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
 * Agglomerative Clustering.
 *
 * @author Rashid Kaleem <rashid.kaleem@gmail.com>
 */

#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/Galois.h"
#include "galois/Reduction.h"

#include "Lonestar/BoilerPlate.h"

#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <fstream>
#include <set>
#include<iostream>
#include<vector>
#include<limits>
#include<set>
#include<map>
#include<stdlib.h>

#include "LeafNode.h"
#include "NodeWrapper.h"
#include "KdTree.h"
#include"ClusterNode.h"
#include <stdlib.h>
#include <sys/time.h>

namespace cll = llvm::cl;

static const char* name = "Unordered Agglomerative Clustering";
static const char* desc = "Clusters data points using the well-known data-mining algorithm";
static const char* url = "agglomerative_clustering";

static cll::opt<int> numPoints("numPoints", cll::desc("Number of Points"), cll::init(1000));

#define DEBUG_CONSOLE 0

using namespace std;

void loopBody(NodeWrapper * cluster, KdTree *kdTree,
		std::vector<NodeWrapper*> * wl, std::vector<ClusterNode*> *clusterArr,
		std::vector<double> * floatArr) ;
///////////////////////////////////////////
void getRandomPoints(vector<LeafNode*> & lights, int numPoints){
	    double dirX = 0;
	    double dirY = 0;
	    double dirZ = 1;
	    AbstractNode::setGlobalMultitime();
	    AbstractNode::setGlobalNumReps();
	    //generating random lights
	    for (int i = 0; i <numPoints; i++) {
	      double x = ((double) rand())/(std::numeric_limits<int>::max());
	      double y = ((double) rand())/(std::numeric_limits<int>::max());
	      double z = ((double) rand())/(std::numeric_limits<int>::max());

	      LeafNode * l = new LeafNode(x, y, z, dirX, dirY, dirZ);
#if	      DEBUG_CONSOLE
	      cout<<"Created "<<*l<<std::endl;
#endif
	      lights[i]=l;
	    }
	    return;
}

////////////////////////////////////////////////////////////
int findMatch(KdTree * tree, NodeWrapper * nodeA,galois::InsertBag<NodeWrapper *> *&newNodes, galois::InsertBag<NodeWrapper *> &allocs,
		vector<double> * coordinatesArray,vector<ClusterNode*> &clusterArray ){
	int addCounter=0;
	if (tree->contains(*nodeA)) {
		NodeWrapper * nodeB = tree->findBestMatch((*nodeA));
		if (nodeB != NULL && tree->contains(*nodeB)) {
			NodeWrapper * nodeBMatch = tree->findBestMatch((*nodeB));
			if (nodeBMatch!=NULL ){
//						cout<<" Match found "<<*nodeA<<" AND " << *nodeB<<endl;
				if(nodeA->equals(*nodeBMatch) && nodeA->equals(*nodeB)==false) {
//							cout<<" A   is "<<*nodeA<<" A-Closes=B:: " << *nodeB<<" B-Closest "<<*nodeBMatch<<endl;
					//Create a new node here.
					if(nodeA<nodeB){
						NodeWrapper * newNode = new NodeWrapper(*nodeA, *nodeB, coordinatesArray,clusterArray);
						newNodes->push(newNode);
						allocs.push(newNode);
						addCounter++;
					}
				}
				else{
					addCounter++;
					newNodes->push(nodeA);
//							cout<<" A   is "<<*nodeA<<" A-Closes=B:: " << *nodeB<<" B-Closest "<<*nodeBMatch<<endl;
				}
			}
		}
		else{
			addCounter++;
			newNodes->push(nodeA);
		}
	}
	return addCounter;
}

/*********************************************************************************************/
void clusterGalois(vector<LeafNode*> & lights) {
	int tempSize = (1 << NodeWrapper::CONE_RECURSE_SIZE) + 1;
	cout << "Temp size is " << tempSize << " coord. arr size should be "<< tempSize * 3 << endl;
	vector<double> * coordinatesArray = new vector<double> (tempSize * 3);
	vector<NodeWrapper*> initialWorklist(lights.size());
	vector<ClusterNode*> clusterArray(tempSize);
	for (unsigned int i = 0; i < lights.size(); i++) {
		NodeWrapper * nw = new NodeWrapper(*(lights[i]));
		initialWorklist[i] = nw;
	}
	KdTree * tree = (KdTree::createTree(initialWorklist));
#if DEBUG_CONSOLE
	cout<<"Tree created "<<*tree<<endl;
	cout<<"================================================================"<<endl;
#endif
	////////////////////////////////////
	vector<NodeWrapper*> workListOld(0);
	KdTree::getAll(*tree, workListOld);
	vector<NodeWrapper*> workList(0);
	size_t size = 0;
	for(unsigned int i=0;i<workListOld.size();i++){
		workList.push_back(workListOld[i]);
	}
  galois::GAccumulator<size_t> addedNodes;
	galois::InsertBag<NodeWrapper *> *newNodes;
	galois::InsertBag<NodeWrapper *> allocs;

	galois::StatTimer T;
	T.start();

	while(true){
		newNodes = new galois::InsertBag<NodeWrapper*>();

		addedNodes.reset();

    using WL = galois::worklists::dChunkedFIFO<16>;

    // find matching
		galois::for_each(galois::iterate(workList), 
        [&] (NodeWrapper* nodeA, auto& lwl) {
          if (tree->contains(*nodeA)) {
              NodeWrapper * nodeB = tree->findBestMatch((*nodeA));
              if (nodeB != NULL && tree->contains(*nodeB)) {
                NodeWrapper * nodeBMatch = tree->findBestMatch((*nodeB));
                if (nodeBMatch!=NULL ){
                  if(nodeA->equals(*nodeBMatch) && nodeA->equals(*nodeB)==false) {
                    //Create a new node here.
                    if(nodeA<nodeB){
                      //galois::Allocator::
                      NodeWrapper * newNode = new NodeWrapper(*nodeA, *nodeB, coordinatesArray,clusterArray);
                      newNodes->push(newNode);
                      allocs.push(newNode);
                      addedNodes +=1;
                    }
                  }
                  else{
                    addedNodes +=1;
                    newNodes->push(nodeA);
                  }
                }
              }
              else{
                addedNodes +=1;
                newNodes->push(nodeA);
              }
          }
        },
        
        
        galois::wl<WL>(), galois::timeit(), galois::loopname("Main"));

		size += addedNodes.reduce();

		workList.clear();
		for(galois::InsertBag<NodeWrapper*>::iterator it = newNodes->begin(), itEnd = newNodes->end();it!=itEnd;it++)
			workList.push_back(*it);
		if(size<2)
			break;
		size=0;
		KdCell::cleanupTree(tree);
		tree = (KdTree::createTree(workList));
		delete newNodes;
	}
	T.stop();
#if DEBUG_CONSOLE
	cout<<"================================================================"<<endl<<*tree<<endl;
#endif

	//!!!!!!!!!!!!!!!!!!!Cleanup
	delete newNodes;
	KdCell::cleanupTree(tree);
	AbstractNode::cleanup();
	for (unsigned int i = 0; i < lights.size(); i++) {
		NodeWrapper * nw = initialWorklist[i];
		delete nw;
	}
	for(galois::InsertBag<NodeWrapper*>::iterator it = allocs.begin(), itEnd = allocs.end();it!=itEnd;it++)
		delete *it;
	delete coordinatesArray;
	return;

}

/*********************************************************************************************/
///////////////////////////////////////////

void clusterSerial(vector<LeafNode*> & lights) {
	int tempSize = (1 << NodeWrapper::CONE_RECURSE_SIZE) + 1;
	cout << "Temp size is " << tempSize << " coord. arr size should be "<< tempSize * 3 << endl;
	vector<double> * coordinatesArray = new vector<double> (tempSize * 3);
	vector<NodeWrapper*> initialWorklist(lights.size());
	vector<ClusterNode*> clusterArray(tempSize);
	for (unsigned int i = 0; i < lights.size(); i++) {
		NodeWrapper * nw = new NodeWrapper(*(lights[i]));
		initialWorklist[i] = nw;
	}
	KdTree * tree = (KdTree::createTree(initialWorklist));
//#if DEBUG_CONSOLE
	cout<<"Tree created "<<*tree<<endl;
	cout<<"================================================================"<<endl;
//#endif
	////////////////////////////////////
	vector<NodeWrapper*> workListOld(0);
	KdTree::getAll(*tree, workListOld);
	vector<NodeWrapper*> workList(0);
	for(unsigned int i=0;i<workListOld.size();i++){
		workList.push_back(workListOld[i]);
	}
	vector<NodeWrapper*> newNodes;
	vector<NodeWrapper*> allocs;
	while(true){
		while (workList.size() > 1) {
			cout << "===================Worklist size :: "<< workList.size() << "===============" << endl;
			NodeWrapper * nodeA = workList.back();
			workList.pop_back();
			if (tree->contains(*nodeA)) {
				NodeWrapper * nodeB = tree->findBestMatch((*nodeA));
				if (nodeB != NULL && tree->contains(*nodeB)) {
					NodeWrapper * nodeBMatch = tree->findBestMatch((*nodeB));
					if (nodeBMatch!=NULL ){
						if(nodeA->equals(*nodeBMatch) && nodeA->equals(*nodeB)==false) {
							if(nodeA<nodeB){
								NodeWrapper * newNode = new NodeWrapper(*nodeA, *nodeB, coordinatesArray,clusterArray);
								newNodes.push_back(newNode);
								allocs.push_back(newNode);
							}
						}
						else{
							newNodes.push_back(nodeA);
						}
					}
				}
				else{
					newNodes.push_back(nodeA);
				}
			}
		}
		workList.clear();
		for(vector<NodeWrapper*>::iterator it = newNodes.begin(), itEnd = newNodes.end();it!=itEnd;it++)
			workList.push_back(*it);
		cout<<"Newly added"<<newNodes.size()<<endl;
		if(newNodes.size()<2)
			break;
		KdCell::cleanupTree(tree);
		tree = (KdTree::createTree(workList));
		newNodes.clear();
	}
#if  DEBUG_CONSOLE
	cout<<"================================================================"<<endl
			<<*tree<<endl;
#endif

	//!!!!!!!!!!!!!!!!!!!Cleanup
	KdCell::cleanupTree(tree);
	AbstractNode::cleanup();
	for (unsigned int i = 0; i < lights.size(); i++) {
		NodeWrapper * nw = initialWorklist[i];
		delete nw;
	}
	for(unsigned int i=0;i<allocs.size();i++)
		delete allocs[i];
	delete coordinatesArray;
	return;

}
///////////////////////////////////////////
int main(int argc, char ** argv){
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);
	std::cout<<"Starting Clustering app...["<<numPoints<<"]"<<std::endl;
	//Initializing...

	vector<LeafNode*> * lights =new vector<LeafNode*>(numPoints);
	getRandomPoints(*lights, numPoints);
//	clusterSerial(*lights);
	clusterGalois(*lights);
	//Cleaning up!
	for(int i=0;i<numPoints;i++){
		LeafNode * l = lights->at(i);
#if DEBUG_CONSOLE
		std::cout<<"deleted :: "<<*l<<std::endl;
#endif
		delete l;
	}
	std::cout<<"Ending Clustering app...["<<lights->size()<<"]"<<std::endl;
	delete lights;
 	return 0;
}
///////////////////////////////////////////
