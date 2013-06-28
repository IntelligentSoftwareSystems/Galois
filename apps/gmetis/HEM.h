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
 * @author Xin Sui <xinsui@cs.utexas.edu>>
 */

#ifndef HEM_H_
#define HEM_H_

class HEMMatcher{
private:
	int maxVertexWeight;
	GGraph* graph;

	MetisGraph* metisGraph;
public:
	HEMMatcher(MetisGraph* metisGraph, int maxVertexWeight) {

		this->metisGraph = metisGraph;
		this->graph=metisGraph->getGraph();
		this->maxVertexWeight=maxVertexWeight;
	}

	bool match(GNode node) {
		MetisNode& nodeData = graph->getData(node, Galois::MethodFlag::NONE);
		if (nodeData.isMatched()) {
			return false;
		}

		GNode matchNode;
		while(true){
			matchNode = node;
			int maxwgt = -1;
			for (GGraph::edge_iterator jj = graph->edge_begin(node, Galois::MethodFlag::NONE), eejj = graph->edge_end(node, Galois::MethodFlag::NONE); jj != eejj; ++jj) {
				GNode neighbor = graph->getEdgeDst(jj);
				MetisNode& neighMNode = graph->getData(neighbor,Galois::MethodFlag::NONE);
				int edgeData = graph->getEdgeData(jj, Galois::MethodFlag::NONE);
				if (!neighMNode.isMatched() && maxwgt < edgeData
						&& nodeData.getWeight() + neighMNode.getWeight() <= maxVertexWeight) {
					maxwgt = edgeData;
					matchNode = neighbor;
				}
			}

			MetisNode& matchNodeData = graph->getData(matchNode, Galois::MethodFlag::CHECK_CONFLICT);
			if(node == matchNode || !matchNodeData.isMatched()){
				//						cout<<(node==matchNode)<<endl;
				break;
			}
		}
		nodeData = graph->getData(node, Galois::MethodFlag::CHECK_CONFLICT);
		/*if(metisGraph->isMatched(nodeData.getNodeId())){
			return;
		}*/
		if(nodeData.isMatched())
			return false;

		MetisNode& matchNodeData = graph->getData(matchNode, Galois::MethodFlag::NONE);
		if(variantMetis::mergeMatching)
			if(nodeData.getNodeId()>matchNodeData.getNodeId()) {
				int id = nodeData.getNodeId();
				nodeData.setNodeId(matchNodeData.getNodeId());
				matchNodeData.setNodeId(id);
			}

#ifdef localNodeData
			nodeData.matchNode = matchNode;
#else
			metisGraph->setMatch(nodeData.getNodeId(), matchNode);
#endif
		nodeData.setMatched(true);

		if (node != matchNode) {
#ifdef localNodeData
				matchNodeData.matchNode = node;
#else
				metisGraph->setMatch(matchNodeData.getNodeId(), node);
#endif
			matchNodeData.setMatched(true);
		}
		return true;


	}

};

#endif /* HEM_H_ */
