/** GMetis -*- C++ -*-
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
 * @author Xin Sui <xinsui@cs.utexas.edu>
 */

#ifndef METISGRAPH_H_
#define METISGRAPH_H_
#include "MetisNode.h"
#include "GMetisConfig.h"
#include <vector>
#include <iostream>
#include "Galois/Atomic.h"
using namespace std;

class MetisGraph{
typedef Galois::GAtomic<int> AtomicInteger;
public:
	MetisGraph(){
		mincut =0;
		numEdges =0;
		numNodes = 0;
		boundaryNodes = NULL;
		coarseGraphMapTo = NULL;
	}

	~MetisGraph() {
		if(boundaryNodes != NULL){
			delete boundaryNodes;
		}
	}

	void initMatches(){
//		matches = new cache_line_storage<GNode>[numNodes];
//		matchFlag = new cache_line_storage<bool>[numNodes];
//		for(int i=0;i<numNodes;i++){
//			matchFlag[i].data = false;
//		}
		matchFlag = new bool[numNodes];
		matches =  new GNode[numNodes];
		std::fill_n(&matchFlag[0], numNodes, false);
//		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
//			GNode node = *ii;
//			matches[node.getData().getNodeId()] = node;
//		}
	}

	void releaseMatches(){
		delete[] matches;
		delete[] matchFlag;
	}

	bool isMatched(int id){
		assert(id < numNodes);
//		return matchFlag[id].data;
		return matchFlag[id];
	}

	GNode getMatch(int id){
		assert(id < numNodes);
//		return matches[id].data;
		return matches[id];
	}

	void initSubGraphMapTo(){
		subGraphMaps = new GNode[numNodes];
	}

	GNode getSubGraphMapTo(int id){
		assert(id < numNodes);
		return subGraphMaps[id];
	}

	void releaseSubGraphMapTo(){
		delete[] subGraphMaps;
	}

	GNode getCoarseGraphMap(int id){
//		return coarseGraphMapTo[id].data;
		return coarseGraphMapTo[id];
	}

	void releaseCoarseGraphMap(){
		if(coarseGraphMapTo!=NULL){
			delete[] coarseGraphMapTo;
			coarseGraphMapTo = NULL;
		}
	}

	void initCoarseGraphMap(){
//		coarseGraphMapTo = new cache_line_storage<GNode>[numNodes];
		coarseGraphMapTo = new GNode[numNodes];
	}

	void setMatch(int id, GNode node){
		assert(id < numNodes);
//		matchFlag[id].data = true;
//		matches[id].data = node;
		matchFlag[id] = true;
		matches[id] = node;
	}

	void setSubGraphMapTo(int id, GNode node){
		assert(id < numNodes);
		 subGraphMaps[id] = node;
	}

	void setCoarseGraphMap(int id, GNode node){
		assert(id < numNodes);
//		coarseGraphMapTo[id].data = node;
		coarseGraphMapTo[id] = node;
	}

	/**
	 * add weight to the weight of a partition
	 * @param index the index of the partition
	 * @param weight the weight to increase
	 * Galois C++ currently does not support abstract lock on integer,
	 * so __sync_fetch_and_add is used to add atomically.
	 * In the future, this will be changed to use abstract lock
	 */
	void incPartWeight(int index, int weight) {
		//__sync_fetch_and_add(&partWeights[index], weight);
		partWeights[index] += weight;
	}

	/**
	 * initialize the partition weights variable
	 */
	void initPartWeight(size_t nparts) {
		if(partWeights.size() != nparts){
			partWeights.resize(nparts);
			for (size_t i = 0; i < partWeights.size(); ++i) {
				partWeights[i] = 0;
			}
		}
	}

	/**
	 * Set the weight of a partition
	 * @param index the index of the partition
	 * @param weight the weight to set
	 */
	void setPartWeight(int index, int weight) {
		partWeights[index]=weight;//.set(weight, MethodFlag.NONE);
	}

	/**
	 * get the weight of a partition
	 * @param part the index of the partition
	 * @return the weight of a partition
	 */
	int getPartWeight(int part) {
		return partWeights[part];
	}


	/**
	 * increase the num of edges by 1 in the graph
	 */
	void incNumEdges() {
		numEdges++;
	}

	/**
	 * return the num of edges in the graph
	 */
	int getNumEdges() {
		return numEdges;
	}

	void setNumEdges(int num) {
		numEdges = num;
	}

	void setNumNodes(int num){
		numNodes = num;
	}
	int getNumNodes(){
		assert(numNodes > 0);
		return numNodes;
	}

	/**
	 * compute the parameters for two-way refining
	 */
	void computeTwoWayPartitionParams() {
		partWeights.resize(2);
		partWeights[0] = 0;
		partWeights[1] = 0;
		//unsetAllBoundaryNodes();
		if(boundaryNodes == NULL){
		  boundaryNodes = new GNodeSet(numNodes, gNodeToInt(graph));
		}else{
			unsetAllBoundaryNodes();
		}
		int mincut = 0;
		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = graph->getData(node,Galois::NONE);
			int me = nodeData.getPartition();
			partWeights[me] += nodeData.getWeight();//set(partWeights[me].get() + nodeData.getWeight(), MethodFlag.NONE);
			updateNodeEdAndId(node);
			if (nodeData.getEdegree() > 0 || (graph->edge_begin(node, Galois::NONE) == graph->edge_end(node,Galois::NONE))) {
				mincut += nodeData.getEdegree();
				setBoundaryNode(node);
			}
		}
		this->mincut =mincut / 2;
	}

	/**
	 * get the maximal adjsum(the sum of the outgoing edge weights of a node) among all the nodes
	 */
	int getMaxAdjSum() {
		int maxAdjSum = -1;
		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			int adjwgtsum = graph->getData(node,Galois::NONE).getAdjWgtSum();
			assert(adjwgtsum>=0||numEdges == 0);
			if (maxAdjSum < adjwgtsum) {
				maxAdjSum = adjwgtsum;
			}
		}
		return maxAdjSum;
	}

	/**
	 * compute the parameters for kway refining
	 */
	void computeKWayPartitionParams(int nparts) {
//		unsetAllBoundaryNodes();
		if(boundaryNodes == NULL){
		  boundaryNodes = new GNodeSet(numNodes, gNodeToInt(graph));
		}else{
			unsetAllBoundaryNodes();
		}
		partWeights.resize(nparts);
		for (int i = 0; i < nparts; ++i) {
			partWeights[i] = 0;
		}
		int mincut = 0;
		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = graph->getData(node,Galois::NONE);
			int me = nodeData.getPartition();
			partWeights[me] +=  nodeData.getWeight();
			updateNodeEdAndId(node);
			if (nodeData.getEdegree() > 0) {
				mincut += nodeData.getEdegree();
				int numEdges = std::distance(graph->edge_begin(node, Galois::NONE), graph->edge_end(node, Galois::NONE));
				nodeData.initPartEdAndIndex(numEdges);

				for (GGraph::edge_iterator jj = graph->edge_begin(node, Galois::NONE), eejj = graph->edge_end(node, Galois::NONE); jj != eejj; ++jj) {
				  GNode neighbor = graph->getEdgeDst(jj);
				  MetisNode& neighborData = graph->getData(neighbor, Galois::NONE);
					if (me != neighborData.getPartition()) {
						int edgeWeight = (int) graph->getEdgeData(jj, Galois::NONE);
						int k = 0;
						for (; k < nodeData.getNDegrees(); k++) {
							if (nodeData.getPartIndex()[k] == neighborData.getPartition()) {
								nodeData.getPartEd()[k] += edgeWeight;
								break;
							}
						}
						if (k == nodeData.getNDegrees()) {
							nodeData.getPartIndex()[nodeData.getNDegrees()] = neighborData.getPartition();
							nodeData.getPartEd()[nodeData.getNDegrees()] = edgeWeight;
							nodeData.setNDegrees(nodeData.getNDegrees() + 1);
						}
					}

				}
			}
			if (nodeData.getEdegree() - nodeData.getIdegree() > 0) {
				setBoundaryNode(node);
			}
		}
		setMinCut(mincut / 2);
	}


	/**
	 * update the external and internal degree for every node in the graph
	 */
	void updateNodeEdAndId(GNode node) {
		int ed = 0;
		int id = 0;
		MetisNode& nodeData = graph->getData(node,Galois::NONE);
		for (GGraph::edge_iterator jj = graph->edge_begin(node, Galois::NONE), eejj = graph->edge_end(node, Galois::NONE); jj != eejj; ++jj) {
		  GNode neighbor = graph->getEdgeDst(jj);
			int weight = (int) graph->getEdgeData(jj, Galois::NONE);
			if (nodeData.getPartition() != graph->getData(neighbor, Galois::NONE).getPartition()) {
				ed = ed + weight;
			} else {
				id = id + weight;
			}
		}
		nodeData.setEdegree(ed);
		nodeData.setIdegree(id);
	}

	/**
	 * return the intgraph in the wrapper
	 */
	GGraph* getGraph() {
		return graph;
	}

	/**
	 * set the graph for the wrapper
	 */
	void setGraph(GGraph* graph) {
		this->graph = graph;
	}

	/**
	 * return the finer metisGraph
	 */
	MetisGraph* getFinerGraph() {
		return finerGraph;
	}

	/**
	 * set the finer metisGraph
	 */
	void setFinerGraph(MetisGraph* finer) {
		finerGraph = finer;
	}

	/**
	 * return the graphcut
	 */
	int getMinCut() {
		return mincut;
	}

	/**
	 * set the graphcut
	 */
	void setMinCut(int cut) {
//		mincut.set(cut);
		mincut = cut;
	}

	/**
	 * increase the graphcut
	 */
	void incMinCut(int cut) {
//		mincut.add(cut, MethodFlag.NONE);
		mincut+=cut;
	}

	//methods for dealing with boundary nodes

	/**
	 * return the number of boundary nodes in the graph
	 */
	int getNumOfBoundaryNodes() {
		return boundaryNodes->size();
	}

	/**
	 * set a node as a boundary node
	 */
	void setBoundaryNode(GNode node) {
	  graph->getData(node, Galois::NONE).setBoundary(true);
//		 pthread_mutex_lock(&mutex);
		boundaryNodes->insert(node);
//		 pthread_mutex_unlock(&mutex);
	}
	//only marks
	void markBoundaryNode(GNode node) {
	  graph->getData(node,Galois::NONE).setBoundary(true);
	}

	//only marks
	void unMarkBoundaryNode(GNode node) {
	  graph->getData(node,Galois::NONE).setBoundary(false);
	}
	/**
	 * unmark a boundary nodes
	 */
	void unsetBoundaryNode(GNode node) {
	  graph->getData(node,Galois::NONE).setBoundary(false);
//		 pthread_mutex_lock(&mutex);

		boundaryNodes->erase(node);
//		 pthread_mutex_unlock(&mutex);
	}

	/**
	 * unset all the boundary nodes
	 */
	void unsetAllBoundaryNodes() {
		for(GNodeSet::iterator iter = boundaryNodes->begin();iter != boundaryNodes->end();++iter){
			GNode node = *iter;
			graph->getData(node,Galois::NONE).setBoundary(false);
		}
		boundaryNodes->clear();
	}

	/**
	 * return the set of boundary nodes
	 */
	GNodeSet* getBoundaryNodes() {
		return boundaryNodes;
	}

    void initBoundarySet(){
    	if(boundaryNodes == NULL){
	  boundaryNodes = new GNodeSet(numNodes, gNodeToInt(graph));
    	}else{
    		unsetAllBoundaryNodes();
    	}
    }

	/**
	 * Compute the sum of the weights of all the outgoing edges for each node in the graph
	 */
	void computeAdjWgtSums() {
		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			graph->getData(node, Galois::NONE).setAdjWgtSum(computeAdjWgtSum(node));
		}
	}

	/**
	 * compute graph cut
	 */
	int computeCut() {
		int cut = 0;
		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			for (GGraph::edge_iterator jj = graph->edge_begin(node, Galois::NONE), eejj = graph->edge_end(node, Galois::NONE); jj != eejj; ++jj) {
			  GNode neighbor = graph->getEdgeDst(jj);
			  if (graph->getData(neighbor,Galois::NONE).getPartition() != graph->getData(node,Galois::NONE).getPartition()) {
					int edgeWeight = (int) graph->getEdgeData(jj, Galois::NONE);
					cut = cut + edgeWeight;
				}
			}
		}

		return cut/2;
	}


	/**
	 * compute the number of edges in the graph
	 */
	int computeEdges() {
		int num = 0;
		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			num += std::distance(graph->edge_begin(node), graph->edge_end(node));
		}
		return num / 2;
	}

	/**
	 * Compute the sum of the weights of all the outgoing edges for a node
	 */
	int computeAdjWgtSum(GNode node) {
		int num = 0;
		for (GGraph::edge_iterator jj = graph->edge_begin(node, Galois::NONE), eejj = graph->edge_end(node, Galois::NONE); jj != eejj; ++jj) {
		  //GNode neighbor = graph->getEdgeDst(jj);
			int weight = (int) graph->getEdgeData(jj, Galois::NONE);
			num = num + weight;
		}
		return num;
	}

	/**
	 * verify if the partitioning is correctly performed by checking
	 * the internal maintained graph cut is same as the real graph cut
	 */
	bool verify() {
		int computedCut = computeCut();
		if (mincut == computedCut) {
			cout<<"mincut is computed correctly:" << mincut <<endl;;
			return true;
		} else {
			cout<<"mincut is computed wrongly:" << mincut << " is not equal " << computedCut <<endl;
			return false;
		}
	}

	/**
	 * check if the partitioning is balanced
	 */
	bool isBalanced(float* tpwgts, float ubfactor) {

		int sum = 0;
		for (size_t i = 0; i < partWeights.size(); i++) {
			sum += partWeights[i];
		}
		for (size_t i = 0; i < partWeights.size(); i++) {
			if (partWeights[i] > tpwgts[i] * sum * (ubfactor + 0.005)) {
				return false;
			}
		}
		return true;
	}

	void computeKWayBalanceBoundary() {
		unsetAllBoundaryNodes();
		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			if (graph->getData(node).getEdegree() > 0) {
				setBoundaryNode(node);
			}
		}

	}

	void computeKWayBoundary() {
		unsetAllBoundaryNodes();
		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = graph->getData(node);
			if (nodeData.getEdegree() - nodeData.getIdegree() >= 0) {
				setBoundaryNode(node);
			}
		}
	}

	float computePartitionBalance(int nparts){
	  	vector<int> kpwgts(nparts);

	  	for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
	  		GNode node = *ii;
	  		kpwgts[graph->getData(node).getPartition()]++;
	  	}
	  	float sumKpwgts=0;
	  	int maxKpwgts=0;
	  	for(int i=0;i<nparts;i++){
	  		sumKpwgts+=kpwgts[i];
	  		if(maxKpwgts < kpwgts[i])
	  			maxKpwgts = kpwgts[i];
	  	}
	  	return nparts * maxKpwgts / sumKpwgts;
	  }


private:
	vector<AtomicInteger> partWeights;
	int mincut;
	int numEdges;
	int numNodes;
	MetisGraph* finerGraph;
	GGraph* graph;
	GNodeSet* boundaryNodes;
//	cache_line_storage<GNode>* matches;
	GNode* matches;
	bool* matchFlag;
//	cache_line_storage<bool>* matchFlag;
	GNode* subGraphMaps;
//	cache_line_storage<GNode>* coarseGraphMapTo;
	GNode* coarseGraphMapTo;
};
#endif /* METISGRAPH_H_ */
