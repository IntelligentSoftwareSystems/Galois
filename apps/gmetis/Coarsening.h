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

#ifndef COARSENING_H_
#define COARSENING_H_
#include "GMetisConfig.h"
#include "HEM.h"
#include "RM.h"
#include <map>
#include <boost/unordered_map.hpp>

using namespace std;
typedef map<int, int> IntIntMap;
typedef vector<int> IntVec;
#include <boost/lexical_cast.hpp>

static const double COARSEN_FRACTION = 0.90;

class Coarsener {
public:

	Coarsener(bool userSerial, int coarsenTo, int maxVertexWeight) {
		this->useSerial = userSerial;
		this->coarsenTo = coarsenTo;
		this->maxVertexWeight = maxVertexWeight;
	}

	int getCoarsenTo() {
		return coarsenTo;
	}

	void createNodes(MetisGraph* coarseMetisGraph){
		GGraph* coarseGraph = coarseMetisGraph->getGraph();
		bool* visited = new bool[metisGraph->getNumNodes()];
		std::fill_n(&visited[0], metisGraph->getNumNodes(), false);
		int id = 0;
		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode nodeData = graph->getData(node,Galois::NONE);
			if(visited[nodeData.getNodeId()]) continue;

			GNode match = metisGraph->getMatch(nodeData.getNodeId());
			MetisNode matchNodeData = graph->getData(match,Galois::NONE);
			int weight = nodeData.getWeight();
			if(match!=node){
				weight+=matchNodeData.getWeight();
			}
			GNode newNode = coarseGraph->createNode(MetisNode(id++, weight));
			coarseGraph->addNode(newNode, Galois::NONE);
			metisGraph->setCoarseGraphMap(nodeData.getNodeId(), newNode);
			if(match!=node){
				metisGraph->setCoarseGraphMap(matchNodeData.getNodeId(), newNode);
			}
			visited[matchNodeData.getNodeId()] = true;
		}
		coarseMetisGraph->setNumNodes(id);
		delete[] visited;
	}

	MetisGraph* coarsen(MetisGraph* metisGraph) {
		bool notFirstTime = false; // use when the graph have all weights equal
		bool* visited = new bool[metisGraph->getNumNodes()];
		int level = 0;
		do {
			metisGraph->initMatches();
			metisGraph->initCoarseGraphMap();
			MetisGraph* coarseMetisGraph = new MetisGraph();
			GGraph* coarser = new GGraph();

			coarseMetisGraph->setGraph(coarser);
			this->graph = metisGraph->getGraph();
			this->metisGraph = metisGraph;
			if (useSerial) {
				notFirstTime = serialMatch(notFirstTime, coarser);
				createNodes(coarseMetisGraph);
				serialCreateCoarserGraph(coarseMetisGraph, visited);
			} else {
				notFirstTime = parallelMatch(notFirstTime, coarser, level);
				createNodes(coarseMetisGraph);
				parallelCreateCoarserGraph(coarseMetisGraph, visited, level++);
			}
			int numEdges = 0;
			for (GGraph::iterator ii = coarser->begin(), ee = coarser->end(); ii != ee; ++ii) {
			  numEdges += std::distance(graph->edge_begin(*ii, Galois::NONE), graph->edge_end(*ii, Galois::NONE));
			}
			metisGraph->releaseMatches();
			coarseMetisGraph->setNumEdges(numEdges / 2);
			coarseMetisGraph->setFinerGraph(metisGraph);
			metisGraph = coarseMetisGraph;

		} while (isDone(metisGraph));
		delete[] visited;
		return metisGraph;
	}
private:
	bool serialMatch(bool notFirstTime, GGraph* coarser) {
		if (notFirstTime) {
			HEMMatcher matcher(metisGraph, coarser, maxVertexWeight);

			for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
				GNode node = *ii;
				matcher.match(node);
			}
		} else {
			RMMatcher matcher(metisGraph, coarser, maxVertexWeight);
			for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
				GNode node = *ii;
				matcher.match(node);
			}
			notFirstTime = true;
		}
		return notFirstTime;
	}

	template<typename MatchingPolicy>
	struct parallelMatchNodes {
		GGraph* coarser;
		MatchingPolicy matcher;
		MetisGraph* metisGraph;
		int maxVertexWeight;
		parallelMatchNodes(MetisGraph* metisGraph, GGraph* coarser, int maxVertexWeight):matcher(metisGraph, coarser, maxVertexWeight){
			this->metisGraph = metisGraph;
			this->coarser = coarser;
			this->maxVertexWeight = maxVertexWeight;
		}
		template<typename Context>
		void operator()(GNode item, Context& lwl) {
			matcher.match(item);
		}
	};

	bool parallelMatch(bool notFirstTime, GGraph* coarser, int level){

		if (notFirstTime) {
			parallelMatchNodes<HEMMatcher> pHEM(metisGraph, coarser, maxVertexWeight);
			Galois::for_each<GaloisRuntime::WorkList::ChunkedLIFO<64, GNode> >(graph->begin(), graph->end(), pHEM, "HEM_Match");
//			vector<GNode> v(graph->begin(), graph->end());
//			std::random_shuffle( v.begin(), v.end() );
//			Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<32, GNode> >(v.begin(), v.end(), pHEM);
		} else {
			parallelMatchNodes<RMMatcher> pRM(metisGraph, coarser, maxVertexWeight);
//			vector<GNode> v(graph->begin(), graph->end());
//			std::random_shuffle( v.begin(), v.end() );
//			Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<32, GNode> >(v.begin(), v.end(), pRM);
			Galois::for_each<GaloisRuntime::WorkList::ChunkedLIFO<64, GNode> >(graph->begin(), graph->end(), pRM, "RM_Match");
			notFirstTime = true;
		}
		return notFirstTime;
	}

	/**
	 * determine if the graph is coarse enough
	 */
	bool isDone(MetisGraph* metisGraph) {
		int size = metisGraph->getNumNodes();//graph->size();
		return size > coarsenTo && size < COARSEN_FRACTION * metisGraph->getFinerGraph()->getNumNodes()//->getGraph()->size()
				&& metisGraph->getNumEdges() > size / 2;
	}

	void addNeighbors(int nodeId, GNode node, GGraph* graph, MetisGraph* coarseMetisGraph, IntVec& lmap) {
		GNode matched = metisGraph->getMatch(nodeId);//.getMatch();
		GNode nodeMapTo = metisGraph->getCoarseGraphMap(nodeId);//node.getData(Galois::NONE).getMapTo();
		GGraph* coarseGraph = coarseMetisGraph->getGraph();
		MetisNode& nodeMapToData = coarseGraph->getData(nodeMapTo, Galois::NONE);
		for (GGraph::edge_iterator jj = graph->edge_begin(node, Galois::NONE), eejj = graph->edge_end(node, Galois::NONE); jj != eejj; ++jj) {
		  GNode neighbor = graph->getEdgeDst(jj);
			if (neighbor == matched) {
				continue;
			}
			int edgeWeight = graph->getEdgeData(jj, Galois::NONE);
			GNode neighborMapTo = metisGraph->getCoarseGraphMap(graph->getData(neighbor,Galois::NONE).getNodeId());//neighbor.getData(Galois::NONE).getMapTo();
			int neighMapToId = coarseGraph->getData(neighborMapTo, Galois::NONE).getNodeId();
//			int& weight = lmap[neighMapToId];
			int pos = -1;
			for(int i=lmap.size()-1;i>=0;i--){
				if(lmap[i]  == neighMapToId){
					pos = i;
				}
			}
			if(pos == -1){
			  coarseGraph->getEdgeData(coarseGraph->addEdge(nodeMapTo, neighborMapTo, Galois::NONE)) = edgeWeight;
				coarseMetisGraph->incNumEdges();
				nodeMapToData.incNumEdges();
				nodeMapToData.addEdgeWeight(edgeWeight);
				lmap.push_back(neighMapToId);
			} else {
			  coarseGraph->getEdgeData(coarseGraph->findEdge(nodeMapTo, neighborMapTo, Galois::NONE), Galois::NONE) += edgeWeight;
				nodeMapToData.addEdgeWeight(edgeWeight);
			}
		}
	}

	void addNeighbors(int nodeId, GNode node, GGraph* graph, MetisGraph* coarseMetisGraph) {
		GNode matched = metisGraph->getMatch(nodeId);//.getMatch();
		GNode nodeMapTo = metisGraph->getCoarseGraphMap(nodeId);//node.getData(Galois::NONE).getMapTo();
		GGraph* coarseGraph = coarseMetisGraph->getGraph();
		MetisNode& nodeMapToData = coarseGraph->getData(nodeMapTo, Galois::NONE);
		for (GGraph::edge_iterator jj = graph->edge_begin(node, Galois::NONE), eejj = graph->edge_end(node, Galois::NONE); jj != eejj; ++jj) {
		  GNode neighbor = graph->getEdgeDst(jj);
			if (neighbor == matched) {
				continue;
			}
			int edgeWeight = graph->getEdgeData(jj, Galois::NONE);
			GNode neighborMapTo = metisGraph->getCoarseGraphMap(graph->getData(neighbor,Galois::NONE).getNodeId());//neighbor.getData(Galois::NONE).getMapTo();
                        GGraph::edge_iterator ff = coarseGraph->findEdge(nodeMapTo, neighborMapTo, Galois::NONE);
                        if (ff == coarseGraph->edge_end(nodeMapTo, Galois::NONE)) {
                          coarseGraph->getEdgeData(coarseGraph->addEdge(nodeMapTo, neighborMapTo, Galois::NONE)) = edgeWeight;
                          nodeMapToData.incNumEdges();
                        } else {
                          coarseGraph->getEdgeData(ff) += edgeWeight;
                        }
			nodeMapToData.addEdgeWeight(edgeWeight);
		}
	}


	void addEdges(int nodeId, GNode node, bool* visited, MetisGraph* coarseMetisGraph) {
		if (visited[nodeId])
			return;
		GNode matched = metisGraph->getMatch(nodeId);
		addNeighbors(nodeId, node, graph, coarseMetisGraph);
		if (matched != node) {
			//matched.map(new buildNeighborClosure(graph, coarseMetisGraph, matched, node, lmap), matched);
		  MetisNode& matchedData = graph->getData(matched, Galois::NONE);
			addNeighbors(matchedData.getNodeId(), matched, graph, coarseMetisGraph);
			visited[matchedData.getNodeId()] = true;
		}
	}

	void serialCreateCoarserGraph(MetisGraph* coarseMetisGraph, bool* visited) {
//		cache_line_storage<bool>* visited = new cache_line_storage<bool>[metisGraph->getNumNodes()];
	  std::fill_n(visited, metisGraph->getNumNodes(),false);
//		for(int i=0;i<metisGraph->getNumNodes();i++){
//			visited[i] = false;
//		}

		for (GGraph::iterator ii = graph->begin(), ee = graph->end(); ii != ee; ++ii) {
			GNode node = *ii;
			addEdges(graph->getData(node,Galois::NONE).getNodeId(), node, visited, coarseMetisGraph);
		}
//		delete[] visited;
	}

	struct parallelAddingEdges {
		MetisGraph* coarseMetisGraph;
//		cache_line_storage<bool>* visited;
		bool* visited;
		MetisGraph* metisGraph;
		GGraph* graph;
		Coarsener* coarsener;
		parallelAddingEdges(MetisGraph* metisGraph, MetisGraph* coarseMetisGraph, Coarsener* coarsener, bool* visited){
			this->coarseMetisGraph = coarseMetisGraph;
			graph = metisGraph->getGraph();
			this->visited = visited;//new bool[graph->size()];
//			arrayFill(visited, metisGraph->getNumNodes(),false);
			for(int i=0;i<metisGraph->getNumNodes();i++){
				visited[i] = false;
			}
			this->metisGraph= metisGraph;
			this->coarsener = coarsener;
		}
		//		 ~parallelAddingEdges(){
		////			 delete[] visited;
		//		 }
		template<typename Context>
		void operator()(GNode item, Context& lwl) {
		  MetisNode& nodeData = graph->getData(item,Galois::NONE);
		  graph->getData(metisGraph->getCoarseGraphMap(nodeData.getNodeId()),Galois::CHECK_CONFLICT);
		  coarsener->addEdges(nodeData.getNodeId(), item, visited, coarseMetisGraph);
		}
	};

	void parallelCreateCoarserGraph(MetisGraph* coarseMetisGraph, bool* visited, int level){
//		cache_line_storage<bool>* visited = new cache_line_storage<bool>[metisGraph->getNumNodes()];
//		Galois::Timer t2;
//		t2.start();
		parallelAddingEdges pae(metisGraph, coarseMetisGraph, this, visited);
		Galois::for_each<GaloisRuntime::WorkList::ChunkedLIFO<32, GNode> >(graph->begin(), graph->end(), pae, "AddNeighbors");
//		t2.stop();
//		cout<<"createTime::"<<t2.get()<<endl;
	}

private:
	int coarsenTo;
	int maxVertexWeight;
	bool useSerial;
	GGraph* graph;
	MetisGraph* metisGraph;
};
#endif /* COARSENING_H_ */
