/*
 * MetisGraph.h
 *
 *  Created on: Jun 13, 2011
 *      Author: xinsui
 */

#ifndef METISGRAPH_H_
#define METISGRAPH_H_
#include "MetisNode.h"
#include "GMetisConfig.h"
#include <vector>

using namespace std;

class MetisGraph{
public:
	MetisGraph() {
		mincut =0;
		numEdges =0;
		numNodes = 0;
	}

	~MetisGraph() {
	}

	void initMatches(){
		matches = new GNode[graph->size()];
		matchFlag = new bool[graph->size()];
		arrayFill(matchFlag, graph->size(), false);
	}

	void releaseMatches(){
		delete[] matches;
		delete[] matchFlag;
	}

	bool isMatched(int id){
		return matchFlag[id];
	}

	GNode getMatch(int id){
		return matches[id];
	}

	void initSubGraphMapTo(){
		subGraphMaps = new GNode[graph->size()];
	}

	GNode getSubGraphMapTo(int id){
		return subGraphMaps[id];
	}

	GNode getCoarseGraphMap(int id){
		return coarseGraphMapTo[id];
	}

	void initCoarseGraphMap(){
		coarseGraphMapTo = new GNode[graph->size()];
	}

	void setMatch(int id, GNode node){
		matchFlag[id] = true;
		matches[id] = node;
	}

	void setSubGraphMapTo(int id, GNode node){
		 subGraphMaps[id] = node;
	}

	void setCoarseGraphMap(int id, GNode node){
		 coarseGraphMapTo[id] = node;
	}

	/**
	 * add weight to the weight of a partition
	 * @param index the index of the partition
	 * @param weight the weight to increase
	 */
	void incPartWeight(int index, int weight) {
//		int oldWeight = partWeights[index];
//		partWeights[index] += weight;
		__sync_fetch_and_add(&partWeights[index], weight);
	}

	/**
	 * initialize the partition weights variable
	 */
	void initPartWeight(int nparts) {
		if(partWeights.size()!=nparts){
			partWeights.resize(nparts);
			for (int i = 0; i < partWeights.size(); i++) {
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
		return numNodes;
	}

	/**
	 * compute the parameters for two-way refining
	 */
	void computeTwoWayPartitionParams() {
		partWeights.resize(2);
		partWeights[0] = 0;
		partWeights[1] = 0;
		unsetAllBoundaryNodes();
		int mincut = 0;
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = node.getData(Galois::Graph::NONE);
			int me = nodeData.getPartition();
			partWeights[me] += nodeData.getWeight();//set(partWeights[me].get() + nodeData.getWeight(), MethodFlag.NONE);
			updateNodeEdAndId(node);
			if (nodeData.getEdegree() > 0 || graph->neighborsSize(node,Galois::Graph::NONE) == 0) {
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
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			int adjwgtsum = node.getData(Galois::Graph::NONE).getAdjWgtSum();
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
		unsetAllBoundaryNodes();
		partWeights.resize(nparts);
		for (int i = 0; i < nparts; i++) {
			partWeights[i] = 0;
		}
		int mincut = 0;
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = node.getData(Galois::Graph::NONE);
			int me = nodeData.getPartition();
			partWeights[me] +=  nodeData.getWeight();
			updateNodeEdAndId(node);
			if (nodeData.getEdegree() > 0) {
				mincut += nodeData.getEdegree();
				int numEdges = graph->neighborsSize(node, Galois::Graph::NONE);
				nodeData.initPartEdAndIndex(numEdges);

				for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE, 0), eejj = graph->neighbor_end(node, Galois::Graph::NONE, 0); jj != eejj; ++jj) {
					GNode neighbor = *jj;
					MetisNode& neighborData = neighbor.getData(Galois::Graph::NONE);
					if (me != neighborData.getPartition()) {
						int edgeWeight = (int) graph->getEdgeData(node, neighbor, Galois::Graph::NONE);
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
		MetisNode& nodeData = node.getData(Galois::Graph::NONE);
		for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE, 0), eejj = graph->neighbor_end(node, Galois::Graph::NONE, 0); jj != eejj; ++jj) {
			GNode neighbor = *jj;
			int weight = (int) graph->getEdgeData(node, neighbor, Galois::Graph::NONE);
			if (nodeData.getPartition() != neighbor.getData(Galois::Graph::NONE).getPartition()) {
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
		return boundaryNodes.size();
	}

	/**
	 * mark a node as a boundary node
	 */
	void setBoundaryNode(GNode node) {
		node.getData(Galois::Graph::NONE).setBoundary(true);
//		 pthread_mutex_lock(&mutex);
		boundaryNodes.insert(node);
//		 pthread_mutex_unlock(&mutex);
	}
	//only marks
	void markBoundaryNode(GNode node) {
		node.getData(Galois::Graph::NONE).setBoundary(true);
	}

	//only marks
	void unMarkBoundaryNode(GNode node) {
		node.getData(Galois::Graph::NONE).setBoundary(false);
	}
	/**
	 * unmark a boundary nodes
	 */
	void unsetBoundaryNode(GNode node) {
		node.getData(Galois::Graph::NONE).setBoundary(false);
//		 pthread_mutex_lock(&mutex);

		boundaryNodes.erase(node);
//		 pthread_mutex_unlock(&mutex);
	}

	/**
	 * unmark all the boundary nodes
	 */
	void unsetAllBoundaryNodes() {
		for(GNodeSet::iterator iter = boundaryNodes.begin();iter != boundaryNodes.end();++iter){
			GNode node = *iter;
			node.getData(Galois::Graph::NONE).setBoundary(false);
		}
		boundaryNodes.clear();
	}

	/**
	 * return the set of boundary nodes
	 */
	GNodeSet* getBoundaryNodes() {
		return &boundaryNodes;
	}

	/**
	 * Compute the sum of the weights of all the outgoing edges for each node in the graph
	 */
	void computeAdjWgtSums() {
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			node.getData(Galois::Graph::NONE).setAdjWgtSum(computeAdjWgtSum(node));
		}
	}

	/**
	 * compute graph cut
	 */
	int computeCut() {
		int cut = 0;
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE, 0), eejj = graph->neighbor_end(node, Galois::Graph::NONE, 0); jj != eejj; ++jj) {
				GNode neighbor = *jj;
				if (neighbor.getData(Galois::Graph::NONE).getPartition() != node.getData(Galois::Graph::NONE).getPartition()) {
					int edgeWeight = (int) graph->getEdgeData(node, neighbor, Galois::Graph::NONE);
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
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			num += graph->neighborsSize(node);
		}
		return num / 2;
	}

	/**
	 * Compute the sum of the weights of all the outgoing edges for a node
	 */
	int computeAdjWgtSum(GNode node) {
		int num = 0;
		for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE, 0), eejj = graph->neighbor_end(node, Galois::Graph::NONE, 0); jj != eejj; ++jj) {
			GNode neighbor = *jj;
			int weight = (int) graph->getEdgeData(node, neighbor, Galois::Graph::NONE);
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
		for (int i = 0; i < partWeights.size(); i++) {
			sum += partWeights[i];
		}
		for (int i = 0; i < partWeights.size(); i++) {
			if (partWeights[i] > tpwgts[i] * sum * (ubfactor + 0.005)) {
				return false;
			}
		}
		return true;
	}

	void computeKWayBalanceBoundary() {
		unsetAllBoundaryNodes();
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			if (node.getData().getEdegree() > 0) {
				setBoundaryNode(node);
			}
		}

	}

	void computeKWayBoundary() {
		unsetAllBoundaryNodes();
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = node.getData();
			if (nodeData.getEdegree() - nodeData.getIdegree() >= 0) {
				setBoundaryNode(node);
			}
		}
	}


private:
	vector<int> partWeights;
	int mincut;
	int numEdges;
	int numNodes;
	MetisGraph* finerGraph;
	GGraph* graph;
	GNodeSet boundaryNodes;
	GNode* matches;
	bool* matchFlag;
	GNode* subGraphMaps;
	GNode* coarseGraphMapTo;
};
#endif /* METISGRAPH_H_ */
