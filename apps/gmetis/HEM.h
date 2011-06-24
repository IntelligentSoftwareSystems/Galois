/*
 * HEM.h
 *
 *  Created on: Jun 16, 2011
 *      Author: xinsui
 */

#ifndef HEM_H_
#define HEM_H_

class HEMMatcher{
private:
	int maxVertexWeight;
	GGraph* graph;
	GGraph* coarseGraph;
	MetisGraph* metisGraph;
public:
	HEMMatcher(MetisGraph* metisGraph, GGraph* coarseGraph, int maxVertexWeight) {
		this->coarseGraph=coarseGraph;
		this->metisGraph = metisGraph;
		this->graph=metisGraph->getGraph();
		this->maxVertexWeight=maxVertexWeight;
	}

	void match(GNode node) {
		MetisNode& nodeData = node.getData(Galois::Graph::CHECK_CONFLICT);
		if (metisGraph->isMatched(nodeData.getNodeId())) {
			return;
		}
		GNode matchNode = node;
		for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::CHECK_CONFLICT, 0), eejj = graph->neighbor_end(node, Galois::Graph::CHECK_CONFLICT, 0); jj != eejj; ++jj) {
			GNode neighbor = *jj;
			MetisNode& neighMNode = neighbor.getData(Galois::Graph::CHECK_CONFLICT);
			if (!metisGraph->isMatched(neighMNode.getNodeId()) && node.getData(Galois::Graph::NONE).getWeight() + neighMNode.getWeight() <= maxVertexWeight) {
				matchNode = neighbor;
				break;
			}
		}

		MetisNode& maxNodeData = matchNode.getData(Galois::Graph::NONE);
//		nodeData.setMatch(matchNode);
//		maxNodeData.setMatch(node);
		metisGraph->setMatch(nodeData.getNodeId(), matchNode);
		metisGraph->setMatch(maxNodeData.getNodeId(), node);

		int weight = nodeData.getWeight();
		if (node != matchNode) {
			weight += maxNodeData.getWeight();
		}

		GNode newNode = coarseGraph->createNode(MetisNode(weight));
		coarseGraph->addNode(newNode);
		metisGraph->setCoarseGraphMap(nodeData.getNodeId(), newNode);
		if(matchNode!=node){
			metisGraph->setCoarseGraphMap(maxNodeData.getNodeId(), newNode);
		}
	}
};

#endif /* HEM_H_ */
