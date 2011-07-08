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
class PMetis{
public:
	PMetis(int coasenTo, int maxVertexWeight):coarsener(true, coasenTo, maxVertexWeight) {
	}

//	/**
//	 * Partition the graph using PMetis
//	 */
//	void partition(MetisGraph* metisGraph, int nparts){
//		int maxVertexWeight = (int) (1.5 * ((metisGraph->getGraph()->size()) / Coarsener::COARSEN_FRACTION));
//		float* totalPartitionWeights = new float[nparts];
//		arrayFill(totalPartitionWeights, nparts, 1 / (float) nparts);
//		metisGraph->computeAdjWgtSums();
//		//    PMetis pmetis = new PMetis(maxVertexWeight, 20);
////		mlevelRecursiveBisection(metisGraph, nparts, totalPartitionWeights, 0, 0);
//	}

	/**
	 * totalPartWeights: This is an array containing "nparts" floating point numbers. For partition i , totalPartitionWeights[i] stores the fraction
	 * of the total weight that should be assigned to it.
	 */
	void mlevelRecursiveBisection(MetisGraph* metisGraph, int nparts, float* totalPartWeights, int tpindex,
			int partStartIndex) {

		GGraph* graph = metisGraph->getGraph();
		int totalVertexWeight = 0;
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			totalVertexWeight += node.getData().getWeight();
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
			for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
				GNode node = *ii;
				assert(node.getData().getPartition()>=0);
				node.getData().setPartition(node.getData().getPartition() + partStartIndex);
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
				for (GGraph::active_iterator ii = subGraphs[0].getGraph()->active_begin(), ee = subGraphs[0].getGraph()->active_end(); ii != ee; ++ii) {
					GNode node = *ii;
					MetisNode& nodeData = node.getData(Galois::Graph::NONE);
					nodeData.setPartition(partStartIndex);
					assert(nodeData.getPartition()>=0);
				}
				mlevelRecursiveBisection(&subGraphs[1], nparts - nparts / 2, totalPartWeights, tpindex + nparts / 2,
						partStartIndex + nparts / 2);
				metisGraph->setMinCut(metisGraph->getMinCut() + subGraphs[1].getMinCut());
			}
			for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
				GNode node = *ii;
				MetisNode& nodeData = node.getData();
				nodeData.setPartition(metisGraph->getSubGraphMapTo(nodeData.getNodeId()).getData().getPartition());
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

		// = new MetisGraph[2];
		//    subGraphs[0] = new MetisGraph();
		//    subGraphs[1] = new MetisGraph();
		subGraphs[0].setGraph(new GGraph());
		subGraphs[1].setGraph(new GGraph());
		metisGraph->initSubGraphMapTo();
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = node.getData();
			assert(nodeData.getPartition()>=0);
			GNode newNode = subGraphs[nodeData.getPartition()].getGraph()->createNode(
					MetisNode(subGraphNodeNum[nodeData.getPartition()], nodeData.getWeight()));
//			nodeData.setSubGraphMap(newNode);
			subGraphs[nodeData.getPartition()].getGraph()->addNode(newNode);
			metisGraph->setSubGraphMapTo(nodeData.getNodeId(), newNode);
			subGraphNodeNum[nodeData.getPartition()]++;
		}

//		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
//			GNode node = *ii;
//			MetisNode& nodeData = node.getData();
//			subGraphs[nodeData.getPartition()].getGraph()->addNode(metisGraph->getSubGraphMapTo(nodeData.getNodeId()));
//		}

		subGraphs[0].setNumNodes(subGraphNodeNum[0]);
		subGraphs[1].setNumNodes(subGraphNodeNum[1]);
		assert(subGraphs[0].getNumNodes() == subGraphNodeNum[0]);
		assert(subGraphs[1].getNumNodes() == subGraphNodeNum[1]);

		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = node.getData();
			int index = nodeData.getPartition();
			GGraph* subGraph = subGraphs[index].getGraph();
			metisGraph->getSubGraphMapTo(nodeData.getNodeId()).getData().setAdjWgtSum(nodeData.getAdjWgtSum());
			for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE), eejj = graph->neighbor_end(node, Galois::Graph::NONE); jj != eejj; ++jj) {
				GNode neighbor = *jj;

				MetisNode& neighborData = neighbor.getData();
				int edgeWeight = graph->getEdgeData(node, jj);
				if (!nodeData.isBoundary() || nodeData.getPartition() == neighborData.getPartition()) {
//					subGraph->addEdge(nodeData.getSubGraphMap(), neighborData.getSubGraphMap(), edgeWeight);
					subGraph->addEdge(metisGraph->getSubGraphMapTo(nodeData.getNodeId()), metisGraph->getSubGraphMapTo(neighborData.getNodeId()), edgeWeight);
				} else {
//					nodeData.getSubGraphMap().getData().setAdjWgtSum(
//							nodeData.getSubGraphMap().getData().getAdjWgtSum() - edgeWeight);
					metisGraph->getSubGraphMapTo(nodeData.getNodeId()).getData().setAdjWgtSum(
							metisGraph->getSubGraphMapTo(nodeData.getNodeId()).getData().getAdjWgtSum() - edgeWeight);

				}
			}
		}
	}
	const static double UB_FACTOR = 1;
private:
	Coarsener coarsener;

};

#endif /* PMETIS_H_ */
