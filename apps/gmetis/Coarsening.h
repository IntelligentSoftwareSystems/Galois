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
 * @author Nikunj Yadav nikunj@cs.utexas.edu
 */

#ifndef COARSENING_H_
#define COARSENING_H_
#include "GMetisConfig.h"
#include "RM.h"
#include "HEM.h"
static const double COARSEN_FRACTION = 0.90;


class Coarsener {
private:
	/*
	 * This section has all the structs that coarsener is going to use.
	 */
	typedef Galois::InsertBag<IteratorPairs> edgeIters;

	template<typename MatchingPolicy>
	struct parallelMatchNodes {
		MatchingPolicy matcher;
		MetisGraph* metisGraph;
		int maxVertexWeight;
		parallelMatchNodes(MetisGraph* metisGraph,int maxVertexWeight):matcher(metisGraph, maxVertexWeight){
			this->metisGraph = metisGraph;
			this->maxVertexWeight = maxVertexWeight;
		}
		template<typename Context>
		void operator()(GNode item, Context& lwl) {
			matcher.match(item);

		}
	};

	struct parallelCreateMultiNodes {
		MetisGraph *finerMetisGraph;
		MetisGraph *coarseMetisGraph;
		GGraph *finerGGraph;
		GGraph *coarseGGraph;
		edgeIters &bag;
		parallelCreateMultiNodes(MetisGraph *fineGraph,MetisGraph *coarseGraph,edgeIters &b):bag(b) {
			this->finerMetisGraph = fineGraph;
			this->finerGGraph = finerMetisGraph->getGraph();
			this->coarseMetisGraph = coarseGraph;
			this->coarseGGraph = coarseMetisGraph->getGraph();
		}

		template<typename Context>
		void operator()(GNode item, Context& lwl) {
			MetisNode &nodeData = finerGGraph->getData(item,Galois::MethodFlag::NONE);
			GNode matchItem = finerMetisGraph->getMatch(nodeData.getNodeId());
			MetisNode &matchNodeData = finerGGraph->getData(matchItem,Galois::MethodFlag::NONE);
			int weight = nodeData.getWeight();
			if(nodeData.getNodeId() > matchNodeData.getNodeId())
				return;

			int id = nodeData.getNodeId();

			if(item!=matchItem) {
				weight+=matchNodeData.getWeight();
			}

			GNode multiNode = coarseGGraph->createNode(MetisNode(id,weight));
			finerMetisGraph->setCoarseGraphMap(id,multiNode);
			finerMetisGraph->setCoarseGraphMap(matchNodeData.getNodeId(),multiNode);
			//bag.push(IteratorPairs(finerGGraph->edge_begin(item),finerGGraph->edge_end(item),finerGGraph->edge_begin(matchItem),finerGGraph->edge_end(matchItem),item));
			coarseGGraph->addNode(multiNode);
		}
	};

	struct parallelPopulateEdgesWithIterators {
		MetisGraph *coarseMetisGraph;
		MetisGraph *finerMetisGraph;
		GGraph *coarseGGraph;
		GGraph *finerGGraph;
		Coarsener *coarsener;
		int const_offset;
		parallelPopulateEdgesWithIterators(MetisGraph *fineGraph,MetisGraph *coarseGraph,Coarsener *coarsener) {
			this->finerMetisGraph = fineGraph;
			this->finerGGraph = finerMetisGraph->getGraph();
			this->coarseMetisGraph = coarseGraph;
			this->coarseGGraph = coarseMetisGraph->getGraph();
			this->coarsener = coarsener;
			const_offset = 50;
		}

		template<typename Context>
		void operator()(IteratorPairs &itpairs, Context& lwl) {
			GNode node = itpairs.node;
			MetisNode &nodeData = finerGGraph->getData(node,Galois::MethodFlag::NONE);
			GNode matchNode = finerMetisGraph->getMatch(nodeData.getNodeId());
			GNode multiNode = finerMetisGraph->getCoarseGraphMap(nodeData.getNodeId());
			int iter=0;
			for(;itpairs.first_start!=itpairs.first_end;itpairs.first_start++) {
				if(iter>=const_offset)
					break;
				iter++;
				GNode neighbor = finerGGraph->getEdgeDst(itpairs.first_start);
				if(neighbor == matchNode)
					continue;
				int weight = finerGGraph->getEdgeData(itpairs.first_start);
				MetisNode &neighborData = finerGGraph->getData(neighbor,Galois::MethodFlag::NONE);
				GNode neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
				GGraph::edge_iterator ff = coarseGGraph->findEdge(multiNode,neighborMultiNode);
				if(ff == coarseGGraph->edge_end(multiNode)) {
					coarsener->pnumEdges+=1;
					coarseGGraph->addEdge(multiNode,neighborMultiNode);
				}
			}

			if(matchNode!=node) {
				iter=0;
				for(;itpairs.second_start!=itpairs.second_end;itpairs.second_start++) {
					if(iter>=const_offset)
						break;
					iter++;
					GNode neighbor = finerGGraph->getEdgeDst(itpairs.second_start);
					if(neighbor == node)
						continue;
					int weight = finerGGraph->getEdgeData(itpairs.second_start);
					MetisNode &neighborData = finerGGraph->getData(neighbor,Galois::MethodFlag::NONE);
					GNode neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
					GGraph::edge_iterator ff = coarseGGraph->findEdge(multiNode,neighborMultiNode);
					if(ff == coarseGGraph->edge_end(multiNode)) {
						coarsener->pnumEdges+=1;
						coarseGGraph->addEdge(multiNode,neighborMultiNode);
					}
				}
			}

			if(itpairs.first_start!=itpairs.first_end || (matchNode!=node && itpairs.second_start!=itpairs.second_end))
				lwl.push(itpairs);

		}
	};

	struct parallelPopulateEdgesWithNodes {
		MetisGraph *coarseMetisGraph;
		MetisGraph *finerMetisGraph;
		GGraph *coarseGGraph;
		GGraph *finerGGraph;
		Coarsener *coarsener;
		parallelPopulateEdgesWithNodes(MetisGraph *fineGraph,MetisGraph *coarseGraph,Coarsener *coarsener) {
			this->finerMetisGraph = fineGraph;
			this->finerGGraph = finerMetisGraph->getGraph();
			this->coarseMetisGraph = coarseGraph;
			this->coarseGGraph = coarseMetisGraph->getGraph();
			this->coarsener = coarsener;
		}

		template<typename Context>
		void operator()(GNode node, Context& lwl) {

			MetisNode &nodeData = finerGGraph->getData(node,Galois::MethodFlag::NONE);
			GNode matchNode = finerMetisGraph->getMatch(nodeData.getNodeId());
			MetisNode &matchNodeData = finerGGraph->getData(matchNode,Galois::MethodFlag::NONE);

			if(nodeData.getNodeId() > matchNodeData.getNodeId())
				return;

			GNode selfMultiNode = finerMetisGraph->getCoarseGraphMap(nodeData.getNodeId());
			MetisNode &selfMultiNodeData = coarseGGraph->getData(selfMultiNode,Galois::MethodFlag::NONE);

			for (GGraph::edge_iterator jj = finerGGraph->edge_begin(node, Galois::MethodFlag::NONE), eejj = finerGGraph->edge_end(node, Galois::MethodFlag::NONE)
					;jj != eejj; ++jj) {
				GNode neighbor = finerGGraph->getEdgeDst(jj);
				MetisNode &neighborData = finerGGraph->getData(neighbor,Galois::MethodFlag::NONE);
				if(neighbor != matchNode) {
					int weight = finerGGraph->getEdgeData(jj);
					GNode neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
					GGraph::edge_iterator ff = coarseGGraph->findEdge(selfMultiNode,neighborMultiNode);
					if(ff == coarseGGraph->edge_end(selfMultiNode)) {
						coarseGGraph->getEdgeData(coarseGGraph->addEdge(selfMultiNode,neighborMultiNode))=weight;
						coarsener->pnumEdges+=1;
						selfMultiNodeData.incNumEdges();
					}
					else {
						//Should Never happen
						coarseGGraph->getEdgeData(ff)+=weight;
					}

					selfMultiNodeData.addEdgeWeight(weight);
				}

			}

			if(matchNode!=node) {

				for (GGraph::edge_iterator jj = finerGGraph->edge_begin(matchNode, Galois::MethodFlag::NONE), eejj = finerGGraph->edge_end(matchNode, Galois::MethodFlag::NONE)
						; jj != eejj; ++jj) {
					GNode neighbor = finerGGraph->getEdgeDst(jj);
					MetisNode &neighborData = finerGGraph->getData(neighbor,Galois::MethodFlag::NONE);
					if(neighbor != node) {
						int weight = finerGGraph->getEdgeData(jj);
						GNode neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
						GGraph::edge_iterator ff = coarseGGraph->findEdge(selfMultiNode,neighborMultiNode);
						if(ff == coarseGGraph->edge_end(selfMultiNode)) {
							coarseGGraph->getEdgeData(coarseGGraph->addEdge(selfMultiNode,neighborMultiNode))=weight;
							coarsener->pnumEdges+=1;
							selfMultiNodeData.incNumEdges();
						}
						else {
							coarseGGraph->getEdgeData(ff)+=weight;
						}
						selfMultiNodeData.addEdgeWeight(weight);
					}

				}

			}
		}
};


void createNodes(MetisGraph* coarseMetisGraph){
	GGraph* coarseGraph = coarseMetisGraph->getGraph();
	MetisGraph *metisGraph = finerMetisGraph;
	bool* visited = new bool[metisGraph->getNumNodes()];
	std::fill_n(&visited[0], metisGraph->getNumNodes(), false);
	int id = 0;
	for (GGraph::iterator ii = finerGGraph->begin(), ee = finerGGraph->end(); ii != ee; ++ii) {
		GNode node = *ii;
		MetisNode nodeData = finerGGraph->getData(node,Galois::MethodFlag::NONE);
		if(visited[nodeData.getNodeId()]) continue;

		GNode match = metisGraph->getMatch(nodeData.getNodeId());
		MetisNode matchNodeData = finerGGraph->getData(match,Galois::MethodFlag::NONE);
		int weight = nodeData.getWeight();
		if(match!=node){
			weight+=matchNodeData.getWeight();
		}
		GNode newNode = coarseGraph->createNode(MetisNode(id, weight));
		id++;
		coarseGraph->addNode(newNode, Galois::MethodFlag::NONE);
		metisGraph->setCoarseGraphMap(nodeData.getNodeId(), newNode);
		if(match!=node){
			metisGraph->setCoarseGraphMap(matchNodeData.getNodeId(), newNode);
		}
		visited[matchNodeData.getNodeId()] = true;
	}
	coarseMetisGraph->setNumNodes(id);
	delete[] visited;
}
public:
Coarsener(bool useSerial,int coarsenTo,int maxVertexWeight) {
	this->useSerial = useSerial;
	this->coarsenTo = coarsenTo;
	this->maxVertexWeight = maxVertexWeight;
	finerMetisGraph = NULL;
	finerGGraph = NULL;
}

void findMatching(bool &firstTime) {
	if(firstTime) {
		parallelMatchNodes	<RMMatcher> pRM(finerMetisGraph,maxVertexWeight);
		Galois::for_each_local<Galois::WorkList::ChunkedLIFO<64, GNode> >(*finerGGraph, pRM, "RM_Match");
		firstTime = false;
	}else {
		parallelMatchNodes	<HEMMatcher> pHEM(finerMetisGraph,maxVertexWeight);
		Galois::for_each_local<Galois::WorkList::ChunkedLIFO<64, GNode> >(*finerGGraph, pHEM, "HEM_Match");
	}
}

void createCoarseGraph(MetisGraph *coarseGraph,edgeIters &bag) {

	parallelCreateMultiNodes pCM(finerMetisGraph,coarseGraph,bag);
	Galois::for_each_local<Galois::WorkList::ChunkedLIFO<64, GNode> >(*finerGGraph, pCM, "Multinode Creation");
	GGraph *coarseGGraph = coarseGraph->getGraph();
	parallelPopulateEdgesWithNodes pPE(finerMetisGraph,coarseGraph,this);
	Galois::for_each_local<Galois::WorkList::ChunkedFIFO<64, GNode> >(*finerGGraph,pPE,"Edge Population");
}

MetisGraph* coarsen(MetisGraph *metisGraph) {
	finerMetisGraph = metisGraph;

	bool firstTime = true;

	do {
		finerGGraph = finerMetisGraph->getGraph();
		MetisGraph *coarseMetisGraph = new MetisGraph();
		GGraph *coarseGGraph = new GGraph();
		coarseMetisGraph->setGraph(coarseGGraph);
		finerMetisGraph->initMatches();
		finerMetisGraph->initCoarseGraphMap();

		findMatching(firstTime);

		edgeIters bag;
		createCoarseGraph(coarseMetisGraph,bag);

		int id = 0;
		for(GGraph::iterator ii=coarseGGraph->begin(),ee=coarseGGraph->end();ii!=ee;ii++){
			MetisNode &nodeData = coarseGGraph->getData(*ii);
			nodeData.setNodeId(id);
			id++;
		}

		if(!testMetis::testCoarsening)
			finerMetisGraph->releaseMatches();

		int coarseNumNodes = std::distance(coarseGGraph->begin(),coarseGGraph->end());
		coarseMetisGraph->setNumNodes(coarseNumNodes);
		coarseMetisGraph->setFinerGraph(finerMetisGraph);
		finerMetisGraph = coarseMetisGraph;
		int numEdges =0;
		numEdges =pnumEdges.reduce();
		finerMetisGraph->setNumEdges(numEdges/2);
		pnumEdges.reset();
		if(!isDone(finerMetisGraph,finerMetisGraph->getNumNodes()))
			break;

	}while(true);


	return finerMetisGraph;
}

int getCoarsenTo() {
	return coarsenTo;
}



private:
bool isDone(MetisGraph* metisGraph,int numNodes) {
	int size = numNodes;//graph->size();
	return size > coarsenTo && size < COARSEN_FRACTION * metisGraph->getFinerGraph()->getNumNodes()//->getGraph()->size()
			&& metisGraph->getNumEdges() > size / 2;
}
Galois::GAccumulator<int> pnumNodes;
Galois::GAccumulator<int> pnumEdges;
bool useSerial;
int coarsenTo;
int maxVertexWeight;
MetisGraph* finerMetisGraph;
GGraph* finerGGraph;

};
#endif /* COARSENING_H_ */
