/*
 * Coarsening.h
 *
 *  Created on: Jun 13, 2011
 *      Author: xinsui
 */

#ifndef COARSENING_H_
#define COARSENING_H_
#include "GMetisConfig.h"
#include "HEM.h"
#include "RM.h"
using namespace boost;
#include <boost/unordered_map.hpp>
typedef unordered_map<int, int> IntIntMap;

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
				notFirstTime = parallelMatch(notFirstTime, coarser);
				// assigning id to coarseGraph
				int id = 0;
				for (GGraph::active_iterator ii = coarser->active_begin(), ee = coarser->active_end(); ii != ee; ++ii) {
					GNode node = *ii;
					node.getData().setNodeId(id++);
				}
				coarseMetisGraph->setNumNodes(id);
				parallelCreateCoarserGraph(coarseMetisGraph);
			}
			metisGraph->releaseMatches();
			coarseMetisGraph->setNumEdges(coarseMetisGraph->getNumEdges() / 2);
			coarseMetisGraph->setFinerGraph(metisGraph);
			metisGraph = coarseMetisGraph;
		} while (isDone(metisGraph));
		return metisGraph;
	}
private:
	bool serialMatch(bool notFirstTime, GGraph* coarser) {
		if (notFirstTime) {
			RMMatcher matcher(metisGraph, coarser, maxVertexWeight);

			for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
				GNode node = *ii;
				matcher.match(node);
			}
		} else {
			HEMMatcher matcher(metisGraph, coarser, maxVertexWeight);
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
		MatchingPolicy* matcher;
		MetisGraph* metisGraph;
		int maxVertexWeight;
		parallelMatchNodes(MetisGraph* metisGraph, GGraph* coarser, int maxVertexWeight){
			this->metisGraph = metisGraph;
			this->coarser = coarser;
			this->maxVertexWeight = maxVertexWeight;
			this->matcher = new MatchingPolicy(metisGraph, coarser, maxVertexWeight);
		}
		~parallelMatchNodes(){
			//			delete matcher;
		}
		template<typename Context>
		void operator()(GNode item, Context& lwl) {
			matcher->match(item);
		}
	};

	bool parallelMatch(bool notFirstTime, GGraph* coarser){

		if (notFirstTime) {
			parallelMatchNodes<HEMMatcher> pHEM(metisGraph, coarser, maxVertexWeight);
			GaloisRuntime::WorkList::ChunkedFIFO<GNode, 64> wl;
			wl.fill_initial(graph->active_begin(), graph->active_end());
			Galois::for_each(wl, pHEM);
		} else {
			parallelMatchNodes<RMMatcher> pRM(metisGraph, coarser, maxVertexWeight);
			GaloisRuntime::WorkList::ChunkedFIFO<GNode, 64> wl;
			wl.fill_initial(graph->active_begin(), graph->active_end());
			Galois::for_each(wl, pRM);
			notFirstTime = true;
		}
		return notFirstTime;
	}

	/**
	 * determine if the graph is coarse enough
	 */
	bool isDone(MetisGraph* metisGraph) {
		GGraph* graph = metisGraph->getGraph();
		int size = metisGraph->getNumNodes();//graph->size();
		return size > coarsenTo && size < COARSEN_FRACTION * metisGraph->getFinerGraph()->getNumNodes()//->getGraph()->size()
				&& metisGraph->getNumEdges() > size / 2;
	}

	void addNeighbors(GNode node, GGraph* graph, MetisGraph* coarseMetisGraph, IntIntMap& lmap) {
		GNode matched = metisGraph->getMatch(node.getData(Galois::Graph::NONE).getNodeId());//.getMatch();
		GNode nodeMapTo = metisGraph->getCoarseGraphMap(node.getData(Galois::Graph::NONE).getNodeId());//node.getData(Galois::Graph::NONE).getMapTo();
		GGraph* coarseGraph = coarseMetisGraph->getGraph();
		for (GGraph::neighbor_iterator jj = graph->neighbor_begin(node, Galois::Graph::NONE, 0), eejj = graph->neighbor_end(node, Galois::Graph::NONE, 0); jj != eejj; ++jj) {
			GNode neighbor = *jj;
			if (neighbor == matched) {
				continue;
			}
			int edgeWeight = graph->getEdgeData(node, neighbor, Galois::Graph::NONE);
			GNode neighborMapTo = metisGraph->getCoarseGraphMap(neighbor.getData(Galois::Graph::NONE).getNodeId());//neighbor.getData(Galois::Graph::NONE).getMapTo();
			int neighMapToId = neighborMapTo.getData(Galois::Graph::NONE).getNodeId();
			if (lmap.find(neighMapToId)==lmap.end()) {
				coarseGraph->addEdge(nodeMapTo, neighborMapTo, edgeWeight, Galois::Graph::NONE);
				coarseMetisGraph->incNumEdges();
				MetisNode& nodeMapToData = nodeMapTo.getData(Galois::Graph::NONE);
				nodeMapToData.incNumEdges();
				nodeMapToData.addEdgeWeight(edgeWeight);
				lmap[neighMapToId] = edgeWeight;
			} else {
				int newEdgeWeight = lmap[neighMapToId];
				lmap[neighMapToId] = edgeWeight + newEdgeWeight;
				coarseGraph->getEdgeData(nodeMapTo, neighborMapTo, Galois::Graph::NONE) += edgeWeight;
				nodeMapTo.getData(Galois::Graph::NONE).addEdgeWeight(edgeWeight);
			}
		}
	}

	void addEdges(GNode node, bool* visited, MetisGraph* coarseMetisGraph) {
		MetisNode& nodeData = node.getData(Galois::Graph::NONE);
		if (visited[nodeData.getNodeId()])
			return;
		IntIntMap lmap;
//		lmap.clear();
		GNode matched = metisGraph->getMatch(nodeData.getNodeId());
		MetisNode& matchedData = matched.getData(Galois::Graph::NONE);
		addNeighbors(node, graph, coarseMetisGraph, lmap);
		if (matched != node) {
			//matched.map(new buildNeighborClosure(graph, coarseMetisGraph, matched, node, lmap), matched);
			addNeighbors(matched, graph, coarseMetisGraph, lmap);
		}
		visited[matchedData.getNodeId()] = true;
	}

	void serialCreateCoarserGraph(MetisGraph* coarseMetisGraph) {
		bool* visited = new bool[graph->size()];
		arrayFill(visited, graph->size(),false);
		for (GGraph::active_iterator ii = graph->active_begin(), ee = graph->active_end(); ii != ee; ++ii) {
			GNode node = *ii;
			addEdges(node, visited, coarseMetisGraph);
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
			arrayFill(visited, graph->size(),false);
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
			for (GGraph::neighbor_iterator jj = graph->neighbor_begin(item, Galois::Graph::CHECK_CONFLICT, 0), eejj = graph->neighbor_end(item, Galois::Graph::CHECK_CONFLICT, 0); jj != eejj; ++jj) {
				GNode neighbor = *jj;
				metisGraph->getCoarseGraphMap(neighbor.getData(Galois::Graph::CHECK_CONFLICT).getNodeId()).getData(Galois::Graph::CHECK_CONFLICT);
			}
			for (GGraph::neighbor_iterator jj = graph->neighbor_begin(matched, Galois::Graph::CHECK_CONFLICT, 0), eejj = graph->neighbor_end(matched, Galois::Graph::CHECK_CONFLICT, 0); jj != eejj; ++jj) {
				GNode neighbor = *jj;
				metisGraph->getCoarseGraphMap(neighbor.getData(Galois::Graph::CHECK_CONFLICT).getNodeId()).getData(Galois::Graph::CHECK_CONFLICT);
			}
			coarsener->addEdges(item, visited, coarseMetisGraph);
		}
	};

	void parallelCreateCoarserGraph(MetisGraph* coarseMetisGraph){
		bool* visited = new bool[metisGraph->getGraph()->size()];
		cout<<"start create graph"<<endl;
		parallelAddingEdges pae(metisGraph, coarseMetisGraph, this, visited);
		GaloisRuntime::WorkList::ChunkedFIFO<GNode, 64> wl;
		wl.fill_initial(graph->active_begin(), graph->active_end());
		Galois::for_each(wl, pae);
		cout<<"finish create graph"<<endl;
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
