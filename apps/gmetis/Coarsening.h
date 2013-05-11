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

	/*
	 * This section has all the structs that Coarsener is going to use.
	 */
private:

	typedef Galois::InsertBag<IteratorPairs> edgeIters;

	struct parallelInitMorphGraph {
		GGraph *graph;
		parallelInitMorphGraph(GGraph *g):graph(g) {
		}
		void operator()(unsigned int tid, unsigned int num) {
			int id = tid;
			for(GGraph::iterator ii = graph->local_begin(),ee=graph->local_end();ii!=ee;ii++) {
				GNode node = *ii;
				MetisNode &nodeData = graph->getData(node);
				nodeData.setNodeId(id);
				id+=num;
			}
		}
	};


	/*
	 *This operator is responsible for matching.
	1. There are two types of matching. Random and Heavy Edge matching 
	2. Random matching picks any random node above a threshold and matches the nodes. RM.h 
	3. Heavy Edge Matching matches the vertex which is connected by the heaviest edge. HEM.h 
	4. This operator can also create the multinode, i.e. the node which is created on combining two matched nodes.  
	5. You can enable/disable 4th by changing variantMetis::mergeMatching
	 */
	template<typename MatchingPolicy>
	struct parallelMatchNodes {
		typedef int tt_does_not_need_parallel_push;
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
		void operator()(GNode item, Galois::UserContext<GNode> &lwl) {

			bool ret = matcher.match(item);
			if(!variantMetis::mergeMatching){
				return;
			}

			MetisNode &nodeData = finerGGraph->getData(item);
			GNode matchItem;
#ifdef localNodeData
			matchItem = static_cast<GNode>(nodeData.matchNode);
#else
			matchItem = finerMetisGraph->getMatch(nodeData.getNodeId());
#endif
			MetisNode &matchNodeData = finerGGraph->getData(matchItem);

			if(nodeData.getNodeId()>matchNodeData.getNodeId())
				return; //Why did we do this ? Because we want to create one node for two matched nodes. So just follow an order

			/*if(!ret)
				return;*/
			int weight = nodeData.getWeight();
			if(item!=matchItem) {
				weight+=matchNodeData.getWeight();
			}
			/*
			 *Hack Alert:
			 * Creation of multinode. Note that the id is the minimum of the two ids.
			 * But this might not serve your needs if you need sequential ids less than no. of nodes
			 */
			GNode multiNode = coarseGGraph->createNode(MetisNode(nodeData.getNodeId(),weight));
			coarseGGraph->addNode(multiNode);
#ifdef localNodeData
			nodeData.multiNode = multiNode;
			matchNodeData.multiNode = multiNode;
#else
			finerMetisGraph->setCoarseGraphMap(nodeData.getNodeId(),multiNode);
			finerMetisGraph->setCoarseGraphMap(matchNodeData.getNodeId(),multiNode);
#endif


		}
	};

	/*
	 *Operator you use if you are not creating the nodes in the matching.
	 */
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
#ifdef localNodeData
			matchItem = static_cast<GNode>(nodeData.matchNode);
#else
			matchItem = finerMetisGraph->getMatch(nodeData.getNodeId());
#endif

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
#ifdef localNodeData
			nodeData.multiNode = multiNode;
			matchNodeData.multiNode = multiNode;
#else
			finerMetisGraph->setCoarseGraphMap(nodeData.getNodeId(),multiNode);
			finerMetisGraph->setCoarseGraphMap(matchNodeData.getNodeId(),multiNode);
#endif
			//bag.push(IteratorPairs(finerGGraph->edge_begin(item),finerGGraph->edge_end(item),finerGGraph->edge_begin(matchItem),finerGGraph->edge_end(matchItem),item));

		}
	};
	/* The reason for creation of this operator is
	 * There might be huge load imbalance with respect to the amount of new memory created by each thread.
	 * So it might make sense to limit the number of allocations you can do per iteration of operator.
	 * Delay the rest of the allocations by pushing the same work in the worklist.
	 * Next time use different offset (to create the rest of the edges). You will need to create the iterator of pairs
	 * of nodes that are going to be connected in the coarse graph when traversing over the finer graph.
	 */
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
#ifdef localNodeData
			multiNode = static_cast<GNode>(nodeData.multiNode);
			matchNode = static_cast<GNode>(nodeData.matchNode);
#else


			matchNode = finerMetisGraph->getMatch(nodeData.getNodeId());
			multiNode = finerMetisGraph->getCoarseGraphMap(nodeData.getNodeId());
#endif
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
#ifdef localNodeData
				neighborMultiNode = static_cast<GNode>(neighborData.multiNode);
#else

				neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
#endif

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
#ifdef localNodeData
					neighborMultiNode = static_cast<GNode>(neighborData.multiNode);
#else
					neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
#endif
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

	/*
	 * This operator is responsible for doing a union find of the edges between matched nodes and populate the edges
	 * in the coarser graph node.
	 */

	struct parallelPopulateEdgesWithNodes {
		typedef int tt_does_not_need_parallel_push;
		//typedef int tt_needs_per_iter_alloc;
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
			//		void operator()(GNode node) {
			MetisNode &nodeData = finerGGraph->getData(node,Galois::MethodFlag::NONE);
			GNode matchNode;
			GNode selfMultiNode;
#ifdef localNodeData
			selfMultiNode = static_cast<GNode>(nodeData.multiNode);
			matchNode = static_cast<GNode>(nodeData.matchNode);

#else
			matchNode = finerMetisGraph->getMatch(nodeData.getNodeId());
			selfMultiNode = finerMetisGraph->getCoarseGraphMap(nodeData.getNodeId());
#endif
			MetisNode &matchNodeData = finerGGraph->getData(matchNode,Galois::MethodFlag::NONE);
			if(nodeData.getNodeId() > matchNodeData.getNodeId())
				return;

			MetisNode &selfMultiNodeData = coarseGGraph->getData(selfMultiNode,Galois::MethodFlag::NONE);
			//#define LC_MORPH_TEST
#ifdef LC_MORPH_TEST
			/*Hack Alert:
			 * This section uses per iteration allocator to make a vector to do union find and then simply create the edges.
			 * It made sense to do this because in the LC_Morph_Graph you need to know how many edges is a node going to have
			 * before hand but not if you have the set of edges you are going to push altogether, and then just push them dynamically
			 * without, checking if a neighbor is already there since you already did that using the vector.
			 */
			int count=0;
			typedef vector<GNode,Galois::PerIterAllocTy::rebind<GNode>::other> svn;
			typedef vector<METISINT,Galois::PerIterAllocTy::rebind<METISINT>::other> svi;
			svn map(lwl.getPerIterAlloc());
			svi mapWeights(lwl.getPerIterAlloc());
			int newWeight=0;
			for (GGraph::edge_iterator jj = finerGGraph->edge_begin(node, Galois::MethodFlag::NONE), eejj = finerGGraph->edge_end(node, Galois::MethodFlag::NONE)
					;jj != eejj; ++jj) {
				GNode neighbor = finerGGraph->getEdgeDst(jj);
				MetisNode &neighborData = finerGGraph->getData(neighbor,Galois::MethodFlag::NONE);

				if(neighbor != matchNode) {
					GNode neighborMultiNode;
					int weight = finerGGraph->getEdgeData(jj);
					if(variantMetis::localNodeData) {
						neighborMultiNode = static_cast<GNode>(neighborData.multiNode);
					}else {
						neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
					}
					bool found=false;
					for(int i=0;i<map.size();i++) {
						if(map[i]==neighborMultiNode){
							found = true;
							mapWeights[i]+=weight;
							break;
						}
					}
					if(!found) {
						count++;
						map.push_back(neighborMultiNode);
						mapWeights.push_back(weight);
						newWeight+=weight;
					}
				}
			}

			if(matchNode!=node)
				for (GGraph::edge_iterator jj = finerGGraph->edge_begin(matchNode, Galois::MethodFlag::NONE), eejj = finerGGraph->edge_end(matchNode, Galois::MethodFlag::NONE)
						;jj != eejj; ++jj) {
					GNode neighbor = finerGGraph->getEdgeDst(jj);
					MetisNode &neighborData = finerGGraph->getData(neighbor,Galois::MethodFlag::NONE);
					if(neighbor != node) {
						GNode neighborMultiNode;
						int weight = finerGGraph->getEdgeData(jj);
						if(variantMetis::localNodeData) {
							neighborMultiNode = static_cast<GNode>(neighborData.multiNode);
						}else {
							neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
						}
						bool found=false;
						for(int i=0;i<map.size();i++) {
							if(map[i]==neighborMultiNode){
								found = true;
								mapWeights[i]+=weight;
								break;
							}
						}
						if(!found) {
							count++;
							map.push_back(neighborMultiNode);
							mapWeights.push_back(weight);
							newWeight+=weight;
						}
					}
				}
			coarseGGraph->addNode(selfMultiNode,count);

			coarsener->pnumEdges+=count;
			selfMultiNodeData.incNumEdges(count);
			selfMultiNodeData.addEdgeWeight(newWeight);
			for(int i=0;i<map.size();i++){
				coarseGGraph->getEdgeData(coarseGGraph->addEdgeWithoutCheck(selfMultiNode,map[i]))=mapWeights[i];
				//coarseGGraph->getEdgeData(coarseGGraph->addEdgeDynamic(selfMultiNode,map[i]))=mapWeights[i];
			}
#else

			for (GGraph::edge_iterator jj = finerGGraph->edge_begin(node, Galois::MethodFlag::NONE), eejj = finerGGraph->edge_end(node, Galois::MethodFlag::NONE)
					;jj != eejj; ++jj) {
				GNode neighbor = finerGGraph->getEdgeDst(jj);
				MetisNode &neighborData = finerGGraph->getData(neighbor,Galois::MethodFlag::NONE);
				if(neighbor != matchNode) {
					int weight = finerGGraph->getEdgeData(jj);
					GNode neighborMultiNode;
#ifdef localNodeData
					neighborMultiNode = static_cast<GNode>(neighborData.multiNode);
#else
					neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
#endif

					GGraph::edge_iterator ff = coarseGGraph->findEdge(selfMultiNode,neighborMultiNode);
					if(ff == coarseGGraph->edge_end(selfMultiNode)) {
#ifdef LC_MORPH
						/*
						 * Hack Alert: This might not the best way to add edges. This will only work if you finish adding the edges of the node in one go
						 * the edge creation can't be interleaved at the thread level. i.e. one thread will create the edges of one node at a time.
						 * You will also see that I dont have to use the addNode(node,number of edges) anymore because of this assumption.
						 */
						coarseGGraph->getEdgeData(coarseGGraph->addEdgeDynamic(selfMultiNode,neighborMultiNode))=weight;
#else
						coarseGGraph->getEdgeData(coarseGGraph->addEdge(selfMultiNode,neighborMultiNode))=weight;
#endif

						coarsener->pnumEdges+=1;
						selfMultiNodeData.incNumEdges();
					}
					else {
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
#ifdef localNodeData
						neighborMultiNode = static_cast<GNode>(neighborData.multiNode);
#else
						neighborMultiNode = finerMetisGraph->getCoarseGraphMap(neighborData.getNodeId());
#endif
						GGraph::edge_iterator ff = coarseGGraph->findEdge(selfMultiNode,neighborMultiNode);
						if(ff == coarseGGraph->edge_end(selfMultiNode)) {
#ifdef LC_MORPH
							coarseGGraph->getEdgeData(coarseGGraph->addEdgeDynamic(selfMultiNode,neighborMultiNode))=weight;
#else
							coarseGGraph->getEdgeData(coarseGGraph->addEdge(selfMultiNode,neighborMultiNode))=weight;
#endif
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
#endif
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
		/*
		 * Different worklist versions tried, dChunkedFIFO 256 works best with LC_MORPH_graph.
		 * Another good type would be Lazy Iter.
		 */
		typedef Galois::WorkList::dChunkedFIFO<256, GNode> WL;
		//typedef Galois::WorkList::ChunkedLIFO<64, GNode> WL;
		//typedef Galois::WorkList::LazyIter<decltype(finerGGraph->local_begin()),false> WL;

		if(firstTime) {
			parallelMatchNodes	<RMMatcher> pRM(finerMetisGraph,coarseMetisGraph,maxVertexWeight);
			Galois::for_each_local<WL>(*finerGGraph, pRM, "RM_Match");
			firstTime = false;
		}else {
			parallelMatchNodes	<HEMMatcher> pHEM(finerMetisGraph,coarseMetisGraph,maxVertexWeight);
			Galois::for_each_local<WL>(*finerGGraph, pHEM, "HEM_Match");
		}

	}

	void createCoarseGraph(MetisGraph *coarseGraph,edgeIters &bag) {

		if(!variantMetis::mergeMatching){
			parallelCreateMultiNodes pCM(finerMetisGraph,coarseGraph,bag);
			Galois::for_each_local<Galois::WorkList::dChunkedFIFO<256, GNode> >(*finerGGraph, pCM, "Multinode Creation");
		}
		parallelPopulateEdgesWithNodes pPE(finerMetisGraph,coarseGraph,this);
		//typedef Galois::WorkList::LazyIter<decltype(finerGGraph->local_begin()),false> WL;
		typedef Galois::WorkList::dChunkedFIFO<256, GNode> WL;
		Galois::for_each_local<WL>(*finerGGraph,pPE,"Edge Population");
		//	Galois::do_all_local(*finerGGraph,pPE,"Edge population");
		//Galois::do_all_local(*finerGGraph,pPE,"Edge Population");
	}


	MetisGraph* coarsen(MetisGraph *metisGraph,bool alignSequential=false) {
		finerMetisGraph = metisGraph;
		if(alignSequential)
			//Galois::on_each(parallelInitMorphGraph(finerMetisGraph->getGraph()));
			assignSeqIds(metisGraph);

		bool firstTime = true;
		do {

			finerGGraph = finerMetisGraph->getGraph();
			MetisGraph *coarseMetisGraph = new MetisGraph();
			GGraph *coarseGGraph = new GGraph();
#ifdef LC_MORPH
			coarseGGraph->initialize();
#endif
			coarseMetisGraph->setGraph(coarseGGraph);
#ifndef localNodeData
			finerMetisGraph->initMatches();
			finerMetisGraph->initCoarseGraphMap();
#endif


			findMatching(firstTime,coarseMetisGraph);

			edgeIters bag;
			createCoarseGraph(coarseMetisGraph,bag);

#ifndef localNodeData
			assignSeqIds(coarseMetisGraph);
#endif
			if(alignSequential) {
				assignSeqIds(coarseMetisGraph);
			}



			if(!testMetis::testCoarsening){
#ifndef localNodeData
				finerMetisGraph->releaseMatches();
#endif
			}

			int coarseNumNodes = std::distance(coarseGGraph->begin(),coarseGGraph->end());
			coarseMetisGraph->setNumNodes(coarseNumNodes);
			coarseMetisGraph->setFinerGraph(finerMetisGraph);

			finerMetisGraph = coarseMetisGraph;
			int numEdges =0;
			numEdges =pnumEdges.reduce();
			pnumEdges.reset();
			//cout<<coarseNumNodes<<" "<<numEdges/2<<endl;
			finerMetisGraph->setNumEdges(numEdges/2);
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
