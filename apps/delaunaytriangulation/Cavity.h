/*
 * Cavity.h
 *
 *  Created on: Jan 25, 2011
 *      Author: xinsui
 */

#ifndef DTCAVITY_H_
#define DTCAVITY_H_

#include "Tuple.h"
#include "Element.h"
#include <set>
#include <vector>
#include <iostream>
using namespace std;
class DTCavity{
	typedef std::set<GNode, Galois::PerIterMem::ItAllocTy::rebind<GNode>::other>::iterator GNodeSetIter;
	
	//typedef std::set<GNode>::iterator GNodeSetIter;
	Graph* graph;
	//std::set<GNode> deletingNodes;
	
	std::deque<GNode, Galois::PerIterMem::ItAllocTy::rebind<GNode>::other> oldNodes;
	std::deque<GNode, Galois::PerIterMem::ItAllocTy::rebind<GNode>::other> connectionNodes;
	std::set<GNode, std::less<GNode>, Galois::PerIterMem::ItAllocTy::rebind<GNode>::other> deletingNodes;
	//std::deque<GNode> oldNodes;
	//std::deque<GNode> connectionNodes;
	GNode node;
	DTTuple tuple;
  Galois::PerIterMem* _cnx;
public:

	DTCavity(Graph* g, GNode& n, DTTuple& t, Galois::PerIterMem* cnx)
	:
   graph(g),
	 oldNodes(cnx->PerIterationAllocator),
	 connectionNodes(cnx->PerIterationAllocator),
	 deletingNodes(std::less<GNode>(), cnx->PerIterationAllocator),
	 node(n),
	 tuple(t)
	{
		_cnx = cnx;
	}
	void build() {
		std::vector<GNode, Galois::PerIterMem::ItAllocTy::rebind<GNode>::other> frontier(_cnx->PerIterationAllocator);
		//std::vector<GNode> frontier;
		frontier.push_back(node);
		while(!frontier.empty()){
			GNode curr = frontier.back();
			frontier.pop_back();
			for (Graph::neighbor_iterator ii = graph->neighbor_begin(curr,Galois::Graph::CHECK_CONFLICT),
					ee = graph->neighbor_end(curr,Galois::Graph::CHECK_CONFLICT);
					ii != ee; ++ii) {
				GNode neighbor = *ii;
				DTElement& neighborElement = neighbor.getData(Galois::Graph::CHECK_CONFLICT);

				if (!graph->containsNode(neighbor) || neighbor == node || deletingNodes.find(neighbor) != deletingNodes.end()) {			
					continue;
				};
				if (neighborElement.getBDim() && neighborElement.inCircle(tuple)) {
					deletingNodes.insert(neighbor);
					frontier.push_back(neighbor);
				} else {
					oldNodes.push_back(curr);
					connectionNodes.push_back(neighbor);
				};
			}
		}
	}

	void update(GNodeVector* newNodes){
		DTElement& nodeData = node.getData(Galois::Graph::NONE);
		nodeData.getTuples().pop_back();		
		//vector<DTElement, Galois::PerIterMem::ItAllocTy::rebind<DTElement>::other> newElements;
		//vector<DTElement*> newElements;
		while(!connectionNodes.empty()){
			GNode neighbor = connectionNodes.front();
			connectionNodes.pop_front();
			GNode oldNode = oldNodes.front();
			oldNodes.pop_front();
			int index = graph->getEdgeData(neighbor, oldNode, Galois::Graph::NONE);
			DTElement& neighborElement = neighbor.getData(Galois::Graph::NONE);
			DTElement e(tuple, neighborElement.getPoint(index), neighborElement.getPoint((index + 1) % 3));
			GNode nnode = graph->createNode(e);
			graph->addNode(nnode, Galois::Graph::CHECK_CONFLICT);
			graph->addEdge(nnode, neighbor, 1, Galois::Graph::CHECK_CONFLICT);
			graph->addEdge(neighbor, nnode, index, Galois::Graph::CHECK_CONFLICT);			
			
			int numNeighborsFound = 0;

			for (GNodeVectorIter iter=newNodes->begin();iter!=newNodes->end();iter++) {
				GNode newNode = *iter;
				bool found = false;
				int indexForNewNode = -1;
				int indexForNode = -1;
				DTElement& newNodeData = newNode.getData(Galois::Graph::NONE);
				if (newNodeData.getPoint(1) == e.getPoint(1)) {
					indexForNewNode = 0;
					indexForNode = 0;
					found = true;
				} else if (newNodeData.getPoint(1) == e.getPoint(2)) {
					indexForNewNode = 0;
					indexForNode = 2;
					found = true;
				} else if (newNodeData.getPoint(2) == e.getPoint(1)) {
					indexForNewNode = 2;
					indexForNode = 0;
					found = true;
				} else if (newNodeData.getPoint(2) == e.getPoint(2)) {
					indexForNewNode = 2;
					indexForNode = 2;
					found = true;
				}

				if (found) {
					graph->addEdge(newNode, nnode, indexForNewNode, Galois::Graph::CHECK_CONFLICT);
					graph->addEdge(nnode, newNode, indexForNode, Galois::Graph::CHECK_CONFLICT);
					numNeighborsFound++;
				}
				if (numNeighborsFound == 2) {
					break;
				}
			}
		

		newNodes->push_back(nnode);
		DTElement& nnode_data = nnode.getData();
		//newElements.push_back(&nnode_data);

		DTElement& oldNodeData = oldNode.getData(Galois::Graph::NONE);
		vector<DTTuple>& tuples = oldNodeData.getTuples();
		if (!tuples.empty()) {		
			std::vector<DTTuple> newTuples;
			for (std::vector<DTTuple>::iterator iter = tuples.begin();iter!=tuples.end();iter++) {
				DTTuple t=*iter;
				if (nnode_data.elementContains(t)) {					
					nnode_data.addTuple(t);
				} else {
					newTuples.push_back(t);
				}
			}
			oldNodeData.getTuples().swap(newTuples);
		}
		}
		deletingNodes.insert(node);
		dispatchTuples(newNodes);
		//dispatchTuples(newNodes);
		GNodeSetIter setIter;
		for (setIter = deletingNodes.begin();setIter != deletingNodes.end(); setIter++) {
			GNode dnode = *setIter;
			dnode.getData(Galois::Graph::NONE).setProcessed();
			graph->removeNode(dnode, Galois::Graph::NONE);
		}
	}

	void dispatchTuples(GNodeVector* newNodes) {
		int size = newNodes->size();
		GNodeSetIter iter;
		for (iter=deletingNodes.begin();iter!=deletingNodes.end();iter++) {
			GNode dnode = *iter;
			std::vector<DTTuple>& tuples = dnode.getData(Galois::Graph::NONE).getTuples();
			if (tuples.empty()) {
				continue;
			}
			while (!tuples.empty()) {
				DTTuple tup = tuples.back();
				tuples.pop_back();
				for (int i = 0; i < size; i++) {
					DTElement& element = (*newNodes)[i].getData(Galois::Graph::NONE);
					if ((element.elementContains(tup))) {
						element.addTuple(tup);
						if (i != 0) {
							GNode newNode = (*newNodes)[i];
							(*newNodes)[i] = (*newNodes)[0];
							(*newNodes)[0] = newNode;
						}
						break;
					}
				}
			}
		}
	}


};

#endif /* CAVITY_H_ */
