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
using namespace boost;
using namespace std;
#include <boost/unordered_map.hpp>
//typedef unordered_map<int, int> IntIntMap;
typedef map<int, int> IntIntMap;
typedef vector<int> IntVec;

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

	MetisGraph* coarsen(MetisGraph* metisGraph) {
		bool notFirstTime = false; // use when the graph have all weights equal

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
				// assigning id to coarseGraph
				int id = 0;
				for (GGraph::active_iterator ii = coarser->active_begin(), ee = coarser->active_end(); ii != ee; ++ii) {
					GNode node = *ii;
					node.getData().setNodeId(id++);
				}
				coarseMetisGraph->setNumNodes(id);
				serialCreateCoarserGraph(coarseMetisGraph);
			} else {
				Galois::Timer t1;
				t1.start();
				notFirstTime = parallelMatch(notFirstTime, coarser);
				t1.stop();
				cout<<"matchTime::"<<t1.get()<<endl;

				// assigning id to coarseGraph
				int id = 0;
				for (GGraph::active_iterator ii = coarser->active_begin(), ee = coarser->active_end(); ii != ee; ++ii) {
					GNode node = *ii;
					node.getData().setNodeId(id++);
				}
				coarseMetisGraph->setNumNodes(id);
				Galois::Timer t2;
				t2.start();
				parallelCreateCoarserGraph(coarseMetisGraph);
				t2.stop();
				cout<<"createTime::"<<t2.get()<<endl;
			}
			int numEdges = 0;
			for (GGraph::active_iterator ii = coarser->active_begin(), ee = coarser->active_end(); ii != ee; ++ii) {
				numEdges += graph->neighborsSize(*ii, Galois::Graph::NONE);
			}
			metisGraph->releaseMatches();
			coarseMetisGraph->setNumEdges(numEdges / 2);
			coarseMetisGraph->setFinerGraph(metisGraph);
			metisGraph = coarseMetisGraph;

		} while (isDone(metisGraph));
		return metisGraph;
	}
private:
	bool serialMatch(bool notFirstTime, GGraph* coarser) {
		if (notFirstTime) {
			HEMMatcher matcher(metisGraph, coarser, maxVertexWeight);

			for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
				GNode node = *ii;
				matcher.match(node);
			}
		} else {
			RMMatcher matcher(metisGraph, coarser, maxVertexWeight);
			for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
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

	bool parallelMatch(bool notFirstTime, GGraph* coarser){

		if (notFirstTime) {
			parallelMatchNodes<HEMMatcher> pHEM(metisGraph, coarser, maxVertexWeight);
			Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<64, GNode> >(graph->active_begin(), graph->active_end(), pHEM);
//			vector<GNode> v(graph->active_begin(), graph->active_end());
//			std::random_shuffle( v.begin(), v.end() );
//			Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<32, GNode> >(v.begin(), v.end(), pHEM);
		} else {
			parallelMatchNodes<RMMatcher> pRM(metisGraph, coarser, maxVertexWeight);
			//(metisGraph->getNumNodes());
//			vector<GNode> v(graph->active_begin(), graph->active_end());
//			std::random_shuffle( v.begin(), v.end() );
//			Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<32, GNode> >(v.begin(), v.end(), pRM);
			Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<64, GNode> >(graph->active_begin(), graph->active_end(), pRM);
			notFirstTime = true;
		}
		return notFirstTime;
	}

	/**
	 * determine if the graph is coarse enough
	 */
	bool isDone(MetisGraph* metisGraph) {
		int size = metisGraph->getNumNodes();//graph->size();
//		cout<<metisGraph->getNumNodes()<<"|"<<metisGraph->getNumEdges()<<"<---"<<metisGraph->getFinerGraph()->getNumNodes()<<"|"<<metisGraph->getNumEdges()<<endl;
		return size > coarsenTo && size < COARSEN_FRACTION * metisGraph->getFinerGraph()->getNumNodes()//->getGraph()->size()
				&& metisGraph->getNumEdges() > size / 2;
	}

	void addNeighbors(int nodeId, GNode node, GGraph* graph, MetisGraph* coarseMetisGraph, IntVec& lmap) {
		//int nodeId = node.getData(Galois::Graph::NONE).getNodeId();
		GNode matched = metisGraph->getMatch(nodeId);//.getMatch();
		GNode nodeMapTo = metisGraph->getCoarseGraphMap(nodeId);//node.getData(Galois::Graph::NONE).getMapTo();
		MetisNode& nodeMapToData = nodeMapTo.getData(Galois::Graph::NONE);
		GGraph* coarseGraph = coarseMetisGraph->getGraph();
		for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE), eejj = graph->neighbor_end(node, Galois::Graph::NONE); jj != eejj; ++jj) {
			GNode neighbor = *jj;
			if (neighbor == matched) {
				continue;
			}
			int edgeWeight = graph->getEdgeData(node, jj, Galois::Graph::NONE);
			GNode neighborMapTo = metisGraph->getCoarseGraphMap(neighbor.getData(Galois::Graph::NONE).getNodeId());//neighbor.getData(Galois::Graph::NONE).getMapTo();
			int neighMapToId = neighborMapTo.getData(Galois::Graph::NONE).getNodeId();
//			int& weight = lmap[neighMapToId];
			int pos = -1;
			for(int i=lmap.size()-1;i>=0;i--){
				if(lmap[i]  == neighMapToId){
					pos = i;
				}
			}
			if(pos == -1){
				coarseGraph->addEdge(nodeMapTo, neighborMapTo, edgeWeight, Galois::Graph::NONE);
				coarseMetisGraph->incNumEdges();
				nodeMapToData.incNumEdges();
				nodeMapToData.addEdgeWeight(edgeWeight);
				lmap.push_back(neighMapToId);
			} else {
				coarseGraph->getEdgeData(nodeMapTo, neighborMapTo, Galois::Graph::NONE) += edgeWeight;
				nodeMapToData.addEdgeWeight(edgeWeight);
			}


//			weight += edgeWeight;
//			coarseGraph->addEdge(nodeMapTo, neighborMapTo, weight, Galois::Graph::NONE);
//			coarseMetisGraph->incNumEdges();
//			nodeMapToData.incNumEdges();
//			nodeMapToData.addEdgeWeight(edgeWeight);

//			if (lmap.find(neighMapToId)==lmap.end()) {
//				coarseGraph->addEdge(nodeMapTo, neighborMapTo, edgeWeight, Galois::Graph::NONE);
//				coarseMetisGraph->incNumEdges();
//				MetisNode& nodeMapToData = nodeMapTo.getData(Galois::Graph::NONE);
//				nodeMapToData.incNumEdges();
//				nodeMapToData.addEdgeWeight(edgeWeight);
//				lmap[neighMapToId] = edgeWeight;
//			} else {
//				int newEdgeWeight = lmap[neighMapToId];
//				lmap[neighMapToId] = edgeWeight + newEdgeWeight;
//				coarseGraph->getEdgeData(nodeMapTo, neighborMapTo, Galois::Graph::NONE) += edgeWeight;
//				nodeMapTo.getData(Galois::Graph::NONE).addEdgeWeight(edgeWeight);
//			}
		}
	}

	void addNeighbors(int nodeId, GNode node, GGraph* graph, MetisGraph* coarseMetisGraph) {
		//int nodeId = node.getData(Galois::Graph::NONE).getNodeId();
		GNode matched = metisGraph->getMatch(nodeId);//.getMatch();
		GNode nodeMapTo = metisGraph->getCoarseGraphMap(nodeId);//node.getData(Galois::Graph::NONE).getMapTo();
		MetisNode& nodeMapToData = nodeMapTo.getData(Galois::Graph::NONE);
		GGraph* coarseGraph = coarseMetisGraph->getGraph();
		for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE), eejj = graph->neighbor_end(node, Galois::Graph::NONE); jj != eejj; ++jj) {
			GNode neighbor = *jj;
			if (neighbor == matched) {
				continue;
			}
			int edgeWeight = graph->getEdgeData(node, jj, Galois::Graph::NONE);
			GNode neighborMapTo = metisGraph->getCoarseGraphMap(neighbor.getData(Galois::Graph::NONE).getNodeId());//neighbor.getData(Galois::Graph::NONE).getMapTo();
//			int neighMapToId = neighborMapTo.getData(Galois::Graph::NONE).getNodeId();
			int& weight = coarseGraph->getOrCreateEdge(nodeMapTo, neighborMapTo, Galois::Graph::NONE);
//			int& weight = coarseGraph->getEdgeData(nodeMapTo, neighborMapTo, Galois::Graph::NONE);

			if(weight == 0){
				weight = edgeWeight;
				nodeMapToData.incNumEdges();
//				coarseMetisGraph->incNumEdges();
			}else{
				weight += edgeWeight;
			}
			nodeMapToData.addEdgeWeight(edgeWeight);
		}
//		cout<<"neighborSize:"<<coarseGraph->neighborsSize(nodeMapTo)<<endl;
	}

	void addNeighbors(int nodeId, GNode node, GGraph* graph, MetisGraph* coarseMetisGraph, IntIntMap& lmap) {
			//int nodeId = node.getData(Galois::Graph::NONE).getNodeId();
			GNode matched = metisGraph->getMatch(nodeId);//.getMatch();
			GNode nodeMapTo = metisGraph->getCoarseGraphMap(nodeId);//node.getData(Galois::Graph::NONE).getMapTo();
			MetisNode& nodeMapToData = nodeMapTo.getData(Galois::Graph::NONE);
			GGraph* coarseGraph = coarseMetisGraph->getGraph();
			for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE), eejj = graph->neighbor_end(node, Galois::Graph::NONE); jj != eejj; ++jj) {
				GNode neighbor = *jj;
				if (neighbor == matched) {
					continue;
				}
				int edgeWeight = graph->getEdgeData(node, jj, Galois::Graph::NONE);
				GNode neighborMapTo = metisGraph->getCoarseGraphMap(neighbor.getData(Galois::Graph::NONE).getNodeId());//neighbor.getData(Galois::Graph::NONE).getMapTo();
//				int neighMapToId = neighborMapTo.getData(Galois::Graph::NONE).getNodeId();
//				int& weight = lmap[neighMapToId];
//				weight += edgeWeight;
//				coarseGraph->addEdge(nodeMapTo, neighborMapTo, weight, Galois::Graph::NONE);
				int& weight = coarseGraph->getOrCreateEdge(nodeMapTo, neighborMapTo, Galois::Graph::NONE);

				if(weight == 0){
					weight = edgeWeight;
					nodeMapToData.incNumEdges();
//					coarseMetisGraph->incNumEdges();
				}else{
					weight += edgeWeight;
				}
				nodeMapToData.addEdgeWeight(edgeWeight);

//				if (lmap.find(neighMapToId)==lmap.end()) {
//					coarseGraph->addEdge(nodeMapTo, neighborMapTo, edgeWeight, Galois::Graph::NONE);
//					coarseMetisGraph->incNumEdges();
//					MetisNode& nodeMapToData = nodeMapTo.getData(Galois::Graph::NONE);
//					nodeMapToData.incNumEdges();
//					nodeMapToData.addEdgeWeight(edgeWeight);
//					lmap[neighMapToId] = edgeWeight;
//				} else {
//					int newEdgeWeight = lmap[neighMapToId];
//					lmap[neighMapToId] = edgeWeight + newEdgeWeight;
//					coarseGraph->getEdgeData(nodeMapTo, neighborMapTo, Galois::Graph::NONE) += edgeWeight;
//					nodeMapTo.getData(Galois::Graph::NONE).addEdgeWeight(edgeWeight);
//				}
			}
	}

	void addEdges(int nodeId, GNode node, bool* visited, MetisGraph* coarseMetisGraph) {
//		MetisNode& nodeData = node.getData(Galois::Graph::NONE);
//		int nodeId = nodeData.getNodeId();
		if (visited[nodeId])
			return;
//		lmap.clear();
		GNode matched = metisGraph->getMatch(nodeId);
		addNeighbors(nodeId, node, graph, coarseMetisGraph);
		if (matched != node) {
			//matched.map(new buildNeighborClosure(graph, coarseMetisGraph, matched, node, lmap), matched);
			MetisNode& matchedData = matched.getData(Galois::Graph::NONE);
			addNeighbors(matchedData.getNodeId(), matched, graph, coarseMetisGraph);
			visited[matchedData.getNodeId()] = true;
		}
		//cout<<"nodeID:___"<<graph->neighborsSize(node,Galois::Graph::NONE)<<"|"<<graph->neighborsSize(matched,Galois::Graph::NONE)<<endl;
//		cout<<"maxNeighborSize:"<<maxNeighborSize<<endl;
//		if(maxNeighborSize>100){
//			IntIntMap lmap;
//
//			addNeighbors(nodeId, node, graph, coarseMetisGraph, lmap);
//			if (matched != node) {
//				//matched.map(new buildNeighborClosure(graph, coarseMetisGraph, matched, node, lmap), matched);
//				MetisNode& matchedData = matched.getData(Galois::Graph::NONE);
//				addNeighbors(matchedData.getNodeId(), matched, graph, coarseMetisGraph, lmap);
//				visited[matchedData.getNodeId()] = true;
//			}
//		}else{
//			IntVec lmap;
//			lmap.reserve(50);
//			addNeighbors(nodeId, node, graph, coarseMetisGraph, lmap);
//			if (matched != node) {
//				//matched.map(new buildNeighborClosure(graph, coarseMetisGraph, matched, node, lmap), matched);
//				MetisNode& matchedData = matched.getData(Galois::Graph::NONE);
//				addNeighbors(matchedData.getNodeId(), matched, graph, coarseMetisGraph, lmap);
//				visited[matchedData.getNodeId()] = true;
//
//		}
//		}
	}

	void serialCreateCoarserGraph(MetisGraph* coarseMetisGraph) {
		bool* visited = new bool[metisGraph->getNumNodes()];
		arrayFill(visited, metisGraph->getNumNodes(),false);

		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			MetisNode& nodeData = node.getData(Galois::Graph::NONE);
			int id = nodeData.getNodeId();
			MetisNode& matchData = metisGraph->getMatch(id).getData(Galois::Graph::NONE);
			addEdges(node.getData(Galois::Graph::NONE).getNodeId(), node, visited, coarseMetisGraph);
		}
		delete[] visited;
	}

	struct parallelAddingEdges {
		MetisGraph* coarseMetisGraph;
		bool* visited;
		MetisGraph* metisGraph;
		GGraph* graph;
		Coarsener* coarsener;
		parallelAddingEdges(MetisGraph* metisGraph, MetisGraph* coarseMetisGraph, Coarsener* coarsener, bool* visited){
			this->coarseMetisGraph = coarseMetisGraph;
			graph = metisGraph->getGraph();
			this->visited = visited;//new bool[graph->size()];
			arrayFill(visited, metisGraph->getNumNodes(),false);
			this->metisGraph= metisGraph;
			this->coarsener = coarsener;
		}
		//		 ~parallelAddingEdges(){
		////			 delete[] visited;
		//		 }
		template<typename Context>
		void operator()(GNode item, Context& lwl) {
			MetisNode& nodeData = item.getData(Galois::Graph::CHECK_CONFLICT);
			GNode matched = metisGraph->getMatch(nodeData.getNodeId());
			matched.getData(Galois::Graph::CHECK_CONFLICT);
			// dummy loops for making cautious
//			for (GGraph::neighbor_iterator jj = graph->neighbor_begin(item, Galois::Graph::CHECK_CONFLICT), eejj = graph->neighbor_end(item, Galois::Graph::CHECK_CONFLICT); jj != eejj; ++jj) {
//				GNode neighbor = *jj;
//				metisGraph->getCoarseGraphMap(neighbor.getData(Galois::Graph::CHECK_CONFLICT).getNodeId()).getData(Galois::Graph::CHECK_CONFLICT);
//			}
//			for (GGraph::neighbor_iterator jj = graph->neighbor_begin(matched, Galois::Graph::CHECK_CONFLICT), eejj = graph->neighbor_end(matched, Galois::Graph::CHECK_CONFLICT); jj != eejj; ++jj) {
//				GNode neighbor = *jj;
//				metisGraph->getCoarseGraphMap(neighbor.getData(Galois::Graph::CHECK_CONFLICT).getNodeId()).getData(Galois::Graph::CHECK_CONFLICT);
//			}
			coarsener->addEdges(nodeData.getNodeId(), item, visited, coarseMetisGraph);
		}
	};

	void parallelCreateCoarserGraph(MetisGraph* coarseMetisGraph){
		bool* visited = new bool[metisGraph->getNumNodes()];
//		cout<<"start create graph"<<endl;
		parallelAddingEdges pae(metisGraph, coarseMetisGraph, this, visited);
		Galois::for_each<GaloisRuntime::WorkList::ChunkedFIFO<32, GNode> >(graph->active_begin(), graph->active_end(), pae);
//		cout<<"finish create graph"<<endl;
		delete[] visited;
	}

public:
	static const double COARSEN_FRACTION = 0.9;
private:
	int coarsenTo;
	int maxVertexWeight;
	bool useSerial;
	GGraph* graph;
	MetisGraph* metisGraph;
};
#endif /* COARSENING_H_ */
