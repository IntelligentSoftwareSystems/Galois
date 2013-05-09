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
		MetisGraph* finerMetisGraph;
		MetisGraph* coarseMetisGraph;
		GGraph *finerGGraph;
		GGraph *coarseGGraph;
		int maxVertexWeight;
		parallelMatchNodes(MetisGraph* metisGraph,MetisGraph* coarseMetisGraph,int maxVertexWeight):matcher(metisGraph, maxVertexWeight){
			this->finerMetisGraph = metisGraph;
			this->coarseMetisGraph = coarseMetisGraph;
			this->coarseGGraph = coarseMetisGraph->getGraph();
			this->finerGGraph = finerMetisGraph->getGraph();
			this->maxVertexWeight = maxVertexWeight;
		}
		template<typename Context>
		void operator()(GNode item, Context& lwl) {

			matcher.match(item);
			if(!variantMetis::mergeMatching){
				return;
			}

			MetisNode &nodeData = finerGGraph->getData(item);
			GNode matchItem;
			if(variantMetis::localNodeData) {
				matchItem = static_cast<GNode>(nodeData.matchNode);
				if(nodeData.matchNode==NULL) {
					cout<<"Error in setting match";
				}else if(matchItem == NULL) {
					cout<<"static cast conversion";
				}
			}else {
				matchItem = finerMetisGraph->getMatch(nodeData.getNodeId());
			}
			MetisNode &matchNodeData = finerGGraph->getData(matchItem);
			if(nodeData.getNodeId()>matchNodeData.getNodeId())
				return;

			int weight = nodeData.getWeight();
			if(item!=matchItem) {
				weight+=matchNodeData.getWeight();
			}
			GNode multiNode = coarseGGraph->createNode(MetisNode(nodeData.getNodeId(),weight));
			coarseGGraph->addNode(multiNode);
			if(variantMetis::localNodeData) {
				nodeData.multiNode = multiNode;
				matchNodeData.multiNode = multiNode;
			} else {
				finerMetisGraph->setCoarseGraphMap(nodeData.getNodeId(),multiNode);
				finerMetisGraph->setCoarseGraphMap(matchNodeData.getNodeId(),multiNode);
			}

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
			GNode matchItem;
			if(variantMetis::localNodeData) {
				matchItem = static_cast<GNode>(nodeData.matchNode);
			}else {
				matchItem = finerMetisGraph->getMatch(nodeData.getNodeId());
			}

			MetisNode &matchNodeData = finerGGraph->getData(matchItem,Galois::MethodFlag::NONE);
			int weight = nodeData.getWeight();
			if(nodeData.getNodeId() > matchNodeData.getNodeId())
				return;

			int id = nodeData.getNodeId();

			if(item!=matchItem) {
				weight+=matchNodeData.getWeight();
			}

			GNode multiNode = coarseGGraph->createNode(MetisNode(id,weight));
			coarseGGraph->addNode(multiNode);
			if(variantMetis::localNodeData) {
				nodeData.multiNode = multiNode;
				matchNodeData.multiNode = multiNode;
			} else {
				finerMetisGraph->setCoarseGraphMap(nodeData.getNodeId(),multiNode);
				finerMetisGraph->setCoarseGraphMap(matchNodeData.getNodeId(),multiNode);
			}
			//bag.push(IteratorPairs(finerGGraph->edge_begin(item),finerGGraph->edge_end(item),finerGGraph->edge_begin(matchItem),finerGGraph->edge_end(matchItem),item));

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
			GNode matchNode;
			GNode multiNode;
			if(variantMetis::localNodeData) {
				multiNode = static_cast<GNode>(nodeData.multiNode);
				matchNode = static_cast<GNode>(nodeData.matchNode);

			} else {
				matchNode = finerMetisGraph->getMatch(nodeData.getNodeId());
				multiNode = finerMetisGraph->getCoarseGraphMap(nodeData.getNodeId());
			}
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
				GNode neighborMultiNode;
				if(variantMetis::localNodeData) {
					neighborMultiNode = static_cast<GNode>(neighborData.multiNode);
				}else {
					neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
				}

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
					GNode neighborMultiNode;
					if(variantMetis::localNodeData) {
						neighborMultiNode = static_cast<GNode>(neighborData.multiNode);
					}else {
						neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
					}
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
			GNode matchNode;
			GNode selfMultiNode;
			if(variantMetis::localNodeData) {
				selfMultiNode = static_cast<GNode>(nodeData.multiNode);
				matchNode = static_cast<GNode>(nodeData.matchNode);

			} else {
				matchNode = finerMetisGraph->getMatch(nodeData.getNodeId());
				selfMultiNode = finerMetisGraph->getCoarseGraphMap(nodeData.getNodeId());
			}
			MetisNode &matchNodeData = finerGGraph->getData(matchNode,Galois::MethodFlag::NONE);


			if(nodeData.getNodeId() > matchNodeData.getNodeId())
				return;

			MetisNode &selfMultiNodeData = coarseGGraph->getData(selfMultiNode,Galois::MethodFlag::NONE);

			for (GGraph::edge_iterator jj = finerGGraph->edge_begin(node, Galois::MethodFlag::NONE), eejj = finerGGraph->edge_end(node, Galois::MethodFlag::NONE)
					;jj != eejj; ++jj) {
				GNode neighbor = finerGGraph->getEdgeDst(jj);
				MetisNode &neighborData = finerGGraph->getData(neighbor,Galois::MethodFlag::NONE);
				if(neighbor != matchNode) {
					int weight = finerGGraph->getEdgeData(jj);
					GNode neighborMultiNode;
					if(variantMetis::localNodeData) {
						neighborMultiNode = static_cast<GNode>(neighborData.multiNode);
					}else {
						neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
					}
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
						GNode neighborMultiNode;
						if(variantMetis::localNodeData) {
							neighborMultiNode = static_cast<GNode>(neighborData.multiNode);
						}else {
							neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
						}
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
			//	GNode match = (GNode)nodeData.getMatchNode();
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

	void findMatching(bool &firstTime,MetisGraph* coarseMetisGraph) {
		typedef Galois::WorkList::dChunkedFIFO<256, GNode> WL;
		//typedef Galois::WorkList::ChunkedLIFO<64, GNode> WL;
		if(firstTime) {
			parallelMatchNodes	<RMMatcher> pRM(finerMetisGraph,coarseMetisGraph,maxVertexWeight);

			Galois::for_each_local<WL>(*finerGGraph, pRM, "RM_Match");
			//Galois::do_all_local(*finerGGraph, pRM, "RM_Match");
			firstTime = false;
		}else {
			parallelMatchNodes	<HEMMatcher> pHEM(finerMetisGraph,coarseMetisGraph,maxVertexWeight);
			Galois::for_each_local<WL>(*finerGGraph, pHEM, "HEM_Match");
			//Galois::do_all_local(*finerGGraph, pHEM, "RM_Match");
		}
	}

	void createCoarseGraph(MetisGraph *coarseGraph,edgeIters &bag) {

		if(!variantMetis::mergeMatching){
			parallelCreateMultiNodes pCM(finerMetisGraph,coarseGraph,bag);
			Galois::for_each_local<Galois::WorkList::ChunkedLIFO<64, GNode> >(*finerGGraph, pCM, "Multinode Creation");
		}
		parallelPopulateEdgesWithNodes pPE(finerMetisGraph,coarseGraph,this);
		Galois::for_each_local<Galois::WorkList::ChunkedFIFO<64, GNode> >(*finerGGraph,pPE,"Edge Population");
	}

	MetisGraph* coarsen(MetisGraph *metisGraph,bool alignSequential=false) {
		finerMetisGraph = metisGraph;
		if(alignSequential)
			assignSeqIds(finerMetisGraph);

		bool firstTime = true;
		do {

			finerGGraph = finerMetisGraph->getGraph();
			MetisGraph *coarseMetisGraph = new MetisGraph();
			GGraph *coarseGGraph = new GGraph();
			coarseMetisGraph->setGraph(coarseGGraph);
			if(!variantMetis::localNodeData) {
				finerMetisGraph->initMatches();
				finerMetisGraph->initCoarseGraphMap();
			}


			Galois::Timer t;
			t.start();
			findMatching(firstTime,coarseMetisGraph);
			t.stop();
			//cout<<"Matching Time "<<t.get()<<" ms "<<endl;
			//break;
			edgeIters bag;
			t.start();
			createCoarseGraph(coarseMetisGraph,bag);
			t.stop();
			//cout<<"Creation Time "<<t.get()<<" ms "<<endl;

			if(!variantMetis::localNodeData || alignSequential) {
				t.start();
				assignSeqIds(coarseMetisGraph);
				t.stop();
				//cout<<"Waste Time "<<t.get()<<" ms "<<endl;
			}


			if(!testMetis::testCoarsening && !variantMetis::localNodeData)
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
	void assignSeqIds(MetisGraph *metisGraph) {
		GGraph *graph = metisGraph->getGraph();
		int id = 0;
		for(GGraph::iterator ii=graph->begin(),ee=graph->end();ii!=ee;ii++){
			MetisNode &nodeData = graph->getData(*ii);
			nodeData.setNodeId(id);
			id++;
		}
	}
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
