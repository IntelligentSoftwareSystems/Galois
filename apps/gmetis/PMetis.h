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

#ifndef PMETIS_H_
#define PMETIS_H_

#include "GMetisConfig.h"
#include "MetisGraph.h"
#include "Coarsening.h"
#include "defs.h"

const static double UB_FACTOR = 1;

class PMetis{
public:
	PMetis(int coasenTo, int maxVertexWeight):coarsener(true, coasenTo, maxVertexWeight) {
	}

public:
	/**
	 * totalPartWeights: This is an array containing "nparts" floating point numbers. For partition i , totalPartitionWeights[i] stores the fraction
	 * of the total weight that should be assigned to it.
	 */
	void mlevelRecursiveBisection(MetisGraph* metisGraph, int nparts, float* totalPartWeights, int tpindex,
			int partStartIndex) {

		GGraph* graph = metisGraph->getGraph();
		int totalVertexWeight = 0;
		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			totalVertexWeight += graph->getData(node).getWeight();
		}

		float vertexWeightRatio = 0;
		for (int i = 0; i < nparts / 2; i++) {
			vertexWeightRatio += totalPartWeights[tpindex + i];
		}
		int bisectionWeights[2];
		bisectionWeights[0] = (int) (totalVertexWeight * vertexWeightRatio);
		bisectionWeights[1] = totalVertexWeight - bisectionWeights[0];

		MetisGraph* mcg = coarsener.coarsen(metisGraph);
		bisection(mcg, bisectionWeights, coarsener.getCoarsenTo());
		refineTwoWay(mcg, metisGraph, bisectionWeights);

		if (nparts <= 2) {
			for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
				GNode node = *ii;
				assert(graph->getData(node).getPartition()>=0);
				graph->getData(node).setPartition(graph->getData(node).getPartition() + partStartIndex);
			}
		} else {
			for (int i = 0; i < nparts / 2; i++) {
				totalPartWeights[i + tpindex] *= (1 / vertexWeightRatio);
			}
			//nparts/2 may not be equal to nparts-nparts/2
			for (int i = 0; i < nparts - nparts / 2; i++) {
				totalPartWeights[i + tpindex + nparts / 2] *= (1 / (1 - vertexWeightRatio));
			}
			MetisGraph* subGraphs = new MetisGraph[2];
			splitGraph(metisGraph, subGraphs);
			if (nparts > 3) {
				mlevelRecursiveBisection(&subGraphs[0], nparts / 2, totalPartWeights, tpindex, partStartIndex);
				mlevelRecursiveBisection(&subGraphs[1], nparts - nparts / 2, totalPartWeights, tpindex + nparts / 2,
						partStartIndex + nparts / 2);
				metisGraph->setMinCut(metisGraph->getMinCut() + subGraphs[0].getMinCut() + subGraphs[1].getMinCut());
			} else if (nparts == 3) {
				for (GGraph::iterator ii = subGraphs[0].getGraph()->begin(), ee = subGraphs[0].getGraph()->end(); ii != ee; ++ii) {
					GNode node = *ii;
					MetisNode& nodeData = subGraphs[0].getGraph()->getData(node,Galois::NONE);
					nodeData.setPartition(partStartIndex);
					assert(nodeData.getPartition()>=0);
				}
				mlevelRecursiveBisection(&subGraphs[1], nparts - nparts / 2, totalPartWeights, tpindex + nparts / 2,
						partStartIndex + nparts / 2);
				metisGraph->setMinCut(metisGraph->getMinCut() + subGraphs[1].getMinCut());
			}
			for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
				GNode node = *ii;
				MetisNode& nodeData = graph->getData(node);
				nodeData.setPartition(graph->getData(metisGraph->getSubGraphMapTo(nodeData.getNodeId())).getPartition());
				assert(nodeData.getPartition()>=0);
			}

			metisGraph->releaseSubGraphMapTo();
			delete subGraphs[0].getGraph();
			delete subGraphs[1].getGraph();
			delete[] subGraphs;
		}
	}

	void splitGraph(MetisGraph* metisGraph, MetisGraph* subGraphs) {
		int subGraphNodeNum[2];
		subGraphNodeNum[0] = 0;
		subGraphNodeNum[1] = 0;
		GGraph* graph = metisGraph->getGraph();

		subGraphs[0].setGraph(new GGraph());
		subGraphs[1].setGraph(new GGraph());
		metisGraph->initSubGraphMapTo();
		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = graph->getData(node);
			assert(nodeData.getPartition()>=0);
			GNode newNode = subGraphs[nodeData.getPartition()].getGraph()->createNode(
					MetisNode(subGraphNodeNum[nodeData.getPartition()], nodeData.getWeight()));
			subGraphs[nodeData.getPartition()].getGraph()->addNode(newNode);
			metisGraph->setSubGraphMapTo(nodeData.getNodeId(), newNode);
			subGraphNodeNum[nodeData.getPartition()]++;
		}

		subGraphs[0].setNumNodes(subGraphNodeNum[0]);
		subGraphs[1].setNumNodes(subGraphNodeNum[1]);
		assert(subGraphs[0].getNumNodes() == subGraphNodeNum[0]);
		assert(subGraphs[1].getNumNodes() == subGraphNodeNum[1]);

		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = graph->getData(node);
			int index = nodeData.getPartition();
			GGraph* subGraph = subGraphs[index].getGraph();
			subGraph->getData(metisGraph->getSubGraphMapTo(nodeData.getNodeId())).setAdjWgtSum(nodeData.getAdjWgtSum());
			assert(subGraph->getData(metisGraph->getSubGraphMapTo(nodeData.getNodeId())).getAdjWgtSum()>=0);
			for (GGraph::edge_iterator jj = graph->edge_begin(node, Galois::NONE), eejj = graph->edge_end(node, Galois::NONE); jj != eejj; ++jj) {
			  GNode neighbor = graph->getEdgeDst(jj);

			  MetisNode& neighborData = graph->getData(neighbor);
				int edgeWeight = graph->getEdgeData(jj);
				if (!nodeData.isBoundary() || nodeData.getPartition() == neighborData.getPartition()) {
				  subGraph->getEdgeData(subGraph->addEdge(metisGraph->getSubGraphMapTo(nodeData.getNodeId()), metisGraph->getSubGraphMapTo(neighborData.getNodeId()))) = edgeWeight;
				} else {
				  subGraph->getData(metisGraph->getSubGraphMapTo(nodeData.getNodeId())).setAdjWgtSum(
														     subGraph->getData(metisGraph->getSubGraphMapTo(nodeData.getNodeId())).getAdjWgtSum() - edgeWeight);

				}
			}
		}
	}
private:
	Coarsener coarsener;

};

#endif /* PMETIS_H_ */
