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

#ifndef RANDOMKWAYREFINER_H_
#define RANDOMKWAYREFINER_H_
#include "MetisGraph.h"
#include "GMetisConfig.h"
#include "defs.h"
#include "Galois/Accumulator.h"

class RandomKwayEdgeRefiner {

public:
	RandomKwayEdgeRefiner(float* tpwgts, int nparts, float ubfactor, int npasses, int ffactor) {
		this->tpwgts = tpwgts;
		this->nparts = nparts;
		this->ubfactor = ubfactor;
		this->npasses = npasses;
		this->ffactor = ffactor;
		minwgts = new int[nparts];
		maxwgts = new int[nparts];
		itpwgts = new int[nparts];
	}
	~RandomKwayEdgeRefiner(){
		delete[] minwgts;
		delete[] maxwgts;
		delete[] itpwgts;
	}
	void refine(MetisGraph* metisGraph){

		int tvwgt = 0;
		for (int i = 0; i < nparts; i++) {
			tvwgt += metisGraph->getPartWeight(i);
		}
		for (int i = 0; i < nparts; i++) {
			itpwgts[i] = (int) (tpwgts[i] * tvwgt);
			maxwgts[i] = (int) (tpwgts[i] * tvwgt * ubfactor);
			minwgts[i] = (int) (tpwgts[i] * tvwgt * (1.0 / ubfactor));
		}

		for (int pass = 0; pass < npasses; pass++) {
			int oldcut = metisGraph->getMinCut();

			PerCPUValue perCPUValues;
			parallelRefine pr(metisGraph, this, &perCPUValues);
			Galois::for_each<GaloisRuntime::WorkList::ChunkedLIFO<32, GNode> >(metisGraph->getBoundaryNodes()->begin(), metisGraph->getBoundaryNodes()->end(), pr, "RandomKWAYRefine");
			metisGraph->incMinCut(perCPUValues.mincutInc.reduce());
			GNodeSTLSet& changedNodes = perCPUValues.changedBndNodes.reduce();
			for(GNodeSTLSet::iterator iter=changedNodes.begin();iter!=changedNodes.end();++iter){
				GNode changed = *iter;
				if(metisGraph->getGraph()->getData(changed).isBoundary()){
					metisGraph->getBoundaryNodes()->insert(changed);
				}else{
					metisGraph->getBoundaryNodes()->erase(changed);
				}
			}
//			GGraph* graph = metisGraph->getGraph();
//			for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
//				GNode node = *ii;
//				if(node.getData().isBoundary()){
//					metisGraph->getBoundaryNodes()->insert(node);
//				}else{
//					metisGraph->getBoundaryNodes()->erase(node);
//				}
//			}
			if (metisGraph->getMinCut() == oldcut) {
				break;
			}
		}
	}

private:

  void refineOneNode(MetisGraph* metisGraph, GNode n, PerCPUValue* perCPUValues) {
		GGraph* graph = metisGraph->getGraph();
		MetisNode& nodeData = graph->getData(n,Galois::CHECK_CONFLICT);
		if (nodeData.getEdegree() >= nodeData.getIdegree()) {
			int from = nodeData.getPartition();
			//TODO
			int from_weight=metisGraph->getPartWeight(from);
			int vwgt = nodeData.getWeight();
			if (nodeData.getIdegree() > 0 && from_weight - vwgt < minwgts[from])
				return;
			int k = 0;
			int to = 0;
			long id = nodeData.getIdegree();
			for (k = 0; k < nodeData.getNDegrees(); k++) {
				long gain = nodeData.getPartEd()[k] - id;
				if (gain < 0)
					continue;
				to = nodeData.getPartIndex()[k];

				if (metisGraph->getPartWeight(to) + vwgt <= maxwgts[to] + ffactor * gain && gain >= 0)
					break;
			}
			if (k == nodeData.getNDegrees())
				return;
			for (int j = k + 1; j < nodeData.getNDegrees(); j++) {
				to = nodeData.getPartIndex()[j];
				int to_weight=metisGraph->getPartWeight(to);
				if ((nodeData.getPartEd()[j] > nodeData.getPartEd()[k] && to_weight + vwgt <= maxwgts[to])
						|| (nodeData.getPartEd()[j] == nodeData.getPartEd()[k]
						                                         && itpwgts[nodeData.getPartIndex()[k]] * to_weight < itpwgts[to]
						                                                                                                 * metisGraph->getPartWeight(nodeData.getPartIndex()[k])))
					k = j;
			}

			to = nodeData.getPartIndex()[k];
			int to_weight=metisGraph->getPartWeight(to);
			int j = 0;
			if (nodeData.getPartEd()[k] - nodeData.getIdegree() > 0)
				j = 1;
			else if (nodeData.getPartEd()[k] - nodeData.getIdegree() == 0) {
				if (from_weight >= maxwgts[from]
				                           || itpwgts[from] * (to_weight + vwgt) < itpwgts[to] * from_weight)
					j = 1;
			}
			if (j == 0)
				return;

			/*
			 * if we got here, we can now move the vertex from 'from' to 'to'
			 */
			//dummy for cautious
			graph->edge_begin(n, Galois::CHECK_CONFLICT);
			graph->edge_end(n, Galois::CHECK_CONFLICT);

			perCPUValues->mincutInc += -(nodeData.getPartEd()[k] - nodeData.getIdegree());
			nodeData.setPartition(to);
			metisGraph->incPartWeight(to, vwgt);
			metisGraph->incPartWeight(from, -vwgt);

			nodeData.setEdegree(nodeData.getEdegree() + nodeData.getIdegree() - nodeData.getPartEd()[k]);
			int temp = nodeData.getIdegree();
			nodeData.setIdegree(nodeData.getPartEd()[k]);
			nodeData.getPartEd()[k] = temp;

			if (nodeData.getPartEd()[k] == 0) {
				nodeData.setNDegrees(nodeData.getNDegrees() - 1);
				nodeData.getPartEd()[k] = nodeData.getPartEd()[nodeData.getNDegrees()];
				nodeData.getPartIndex()[k] = nodeData.getPartIndex()[nodeData.getNDegrees()];
			} else {
				nodeData.getPartIndex()[k] = from;
			}

			if (nodeData.getEdegree() - nodeData.getIdegree() < 0){
//				metisGraph->unsetBoundaryNode(n);
				metisGraph->unMarkBoundaryNode(n);
//				perCPUValues->get().changedBndNodes.push_back(n);
				perCPUValues->changedBndNodes.update(n);
			}

			/*
			 * update the degrees of adjacent vertices
			 */
			for (GGraph::edge_iterator jj = graph->edge_begin(n, Galois::NONE), eejj = graph->edge_end(n, Galois::NONE); jj != eejj; ++jj) {
			  GNode neighbor = graph->getEdgeDst(jj);
			  MetisNode& neighborData = graph->getData(neighbor,Galois::NONE);
				if (neighborData.getPartEd().size() == 0) {
					int numEdges = neighborData.getNumEdges();
//					neighborData.partIndex = new int[numEdges];
//					neighborData.partEd = new int[numEdges];
//					cout<<"init"<<endl;
					neighborData.initPartEdAndIndex(numEdges);
				}
				int edgeWeight = graph->getEdgeData(jj, Galois::NONE);
				if (neighborData.getPartition() == from) {
					neighborData.setEdegree(neighborData.getEdegree() + edgeWeight);
					neighborData.setIdegree(neighborData.getIdegree() - edgeWeight);
					if (neighborData.getEdegree() - neighborData.getIdegree() >= 0 && !neighborData.isBoundary())
					{
//						metisGraph->setBoundaryNode(neighbor);
						metisGraph->markBoundaryNode(neighbor);
//						perCPUValues->get().changedBndNodes.push_back(neighbor);
						perCPUValues->changedBndNodes.update(neighbor);
					}
				} else if (neighborData.getPartition() == to) {
					neighborData.setEdegree(neighborData.getEdegree() - edgeWeight);
					neighborData.setIdegree(neighborData.getIdegree() + edgeWeight);
					if (neighborData.getEdegree() - neighborData.getIdegree() < 0 && neighborData.isBoundary())
					{
//						metisGraph->unsetBoundaryNode(neighbor);
						metisGraph->unMarkBoundaryNode(neighbor);
//						perCPUValues->get().changedBndNodes.push_back(neighbor);
						perCPUValues->changedBndNodes.update(neighbor);
					}

				}
				/*Remove contribution from the .ed of 'from' */
				if (neighborData.getPartition() != from) {
					for (int i = 0; i < neighborData.getNDegrees(); i++) {
						if (neighborData.getPartIndex()[i] == from) {
							if (neighborData.getPartEd()[i] == edgeWeight) {
								neighborData.setNDegrees(neighborData.getNDegrees() - 1);
								neighborData.getPartEd()[i] = neighborData.getPartEd()[neighborData.getNDegrees()];
								neighborData.getPartIndex()[i] = neighborData.getPartIndex()[neighborData.getNDegrees()];
							} else {
								neighborData.getPartEd()[i] -= edgeWeight;
							}
							break;
						}
					}
				}
				/*
				 * add contribution to the .ed of 'to'
				 */
				if (neighborData.getPartition() != to) {
					int i;
					for (i = 0; i < neighborData.getNDegrees(); i++) {
						if (neighborData.getPartIndex()[i] == to) {
							neighborData.getPartEd()[i] += edgeWeight;
							break;
						}
					}
					if (i == neighborData.getNDegrees()) {
						int nd = neighborData.getNDegrees();
						neighborData.getPartIndex()[nd] = to;
						neighborData.getPartEd()[nd++] = edgeWeight;
						neighborData.setNDegrees(nd);
					}
				}
			}
		}

	}


	struct parallelRefine {
		MetisGraph* metisGraph;
		RandomKwayEdgeRefiner* refiner;
	  PerCPUValue* perCPUValues;
	  parallelRefine(MetisGraph* metisGraph, RandomKwayEdgeRefiner* refiner, PerCPUValue* perCPUValues){
			this->metisGraph = metisGraph;
			this->refiner = refiner;
			this->perCPUValues = perCPUValues;
		}
		template<typename Context>
		void operator()(GNode item, Context& lwl) {
//			if(perCPUValues->get().changedBndNodes.size() == 0){
//				perCPUValues->get().changedBndNodes.reserve(metisGraph->getNumOfBoundaryNodes());
//				cout<<"resize to "<<metisGraph->getNumOfBoundaryNodes()<<endl;
//			}
		  refiner->refineOneNode(metisGraph, item, perCPUValues);
		}
	};

	float* tpwgts;
	float ubfactor;
	float npasses;
	int ffactor;
	int nparts;
	int* minwgts;
	int* maxwgts;
	int* itpwgts;
};
#endif /* RANDOMKWAYREFINER_H_ */
