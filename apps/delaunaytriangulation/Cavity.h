/** A cavity -*- C++ -*-
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
#ifndef CAVITY_H_
#define CAVITY_H_

#include "Tuple.h"
#include "Element.h"
#include <set>
#include <vector>

class Cavity {
  typedef Galois::PerIterAllocTy::rebind<GNode>::other PerIterGNodeAlloc;
  typedef std::set<GNode, std::less<GNode>, PerIterGNodeAlloc> GNodeSet;
  typedef std::deque<GNode, PerIterGNodeAlloc> GNodeDeque;

  Graph* graph;

  GNodeSet oldNodes;
  GNodeSet deletingNodes;
  GNode node;
  Tuple tuple;
  Galois::PerIterAllocTy& _cnx;

public:
  typedef std::vector<GNode, PerIterGNodeAlloc> GNodeVector;
  
 Cavity(Graph* g, GNode& n, Tuple& t, Galois::PerIterAllocTy& cnx):
   graph(g),
   oldNodes(std::less<GNode>(), cnx),
   deletingNodes(std::less<GNode>(), cnx),
   node(n),
   tuple(t),
   _cnx(cnx)
  {}

  void build() {
    GNodeDeque frontier(_cnx);
    //std::vector<GNode> frontier;
    frontier.push_back(node);
    while (!frontier.empty()){
      GNode curr = frontier.back();
      frontier.pop_back();
      for (Graph::neighbor_iterator ii = graph->neighbor_begin(curr, Galois::CHECK_CONFLICT),
          ee = graph->neighbor_end(curr, Galois::CHECK_CONFLICT);
          ii != ee; ++ii) {
        GNode neighbor = *ii;
        Element& neighborElement = neighbor.getData(Galois::CHECK_CONFLICT);

        if (!graph->containsNode(neighbor) || neighbor == node || deletingNodes.find(neighbor) != deletingNodes.end()) {
          continue;
        }
        if (neighborElement.getBDim() && neighborElement.inCircle(tuple)) {
          deletingNodes.insert(neighbor);
          frontier.push_back(neighbor);
        } else {
          oldNodes.insert(curr);
        }
      }
    }
  }

  void update(GNodeVector* newNodes) {
    Element& nodeData = node.getData(Galois::NONE);
    nodeData.getTuples().pop_back();
    //vector<Element, Galois::PerIterMem::ItAllocTy::rebind<Element>::other> newElements;
    //vector<Element*> newElements;
    for (GNodeSet::iterator it=oldNodes.begin(); it != oldNodes.end(); it++) {
      GNode oldNode = *it;
      for (Graph::neighbor_iterator ii = graph->neighbor_begin(oldNode, Galois::NONE), ee = graph->neighbor_end(oldNode, Galois::NONE); ii != ee; ++ii) {
	GNode neighbor = *ii;
	Element& neighborElement = neighbor.getData(Galois::NONE);
	if (!neighborElement.getBDim() || !neighborElement.inCircle(tuple)) {
	  // Process neighbor
	  int index = graph->getEdgeData(neighbor, oldNode, Galois::NONE);
	  //Element& neighborElement = neighbor.getData(Galois::NONE);
	  Element e(tuple, neighborElement.getPoint(index), neighborElement.getPoint((index + 1) % 3));
	  GNode nnode = graph->createNode(e);
	  graph->addNode(nnode, Galois::NONE);
	  graph->addEdge(nnode, neighbor, 1, Galois::NONE);
	  graph->addEdge(neighbor, nnode, index, Galois::NONE);
	  
	  newNodes->push_back(nnode);
	  Element& nnode_data = nnode.getData(Galois::NONE);
	  //newElements.push_back(&nnode_data);
	  
	  Element& oldNodeData = oldNode.getData(Galois::NONE);
	  TupleList& ntuples = nnode_data.getTuples();
	  TupleList& tuples = oldNodeData.getTuples();
	  if (!tuples.empty()) {
	    TupleList newTuples;
	    for(TupleList::iterator list_iter = tuples.begin(); list_iter != tuples.end(); ++list_iter) {
		Tuple t=*list_iter;
		if (nnode_data.elementContains(t)) {
		  // nnode_data.addTuple(t);
		  ntuples.push_back(t);
		} else {
		  newTuples.push_back(t);
		}
	      }
	    
	    tuples.swap(newTuples);
	  }
	}
      }
    }

    for (unsigned i=0; i<newNodes->size(); i++) {
      GNode n1 = (*newNodes)[i];
      Element& newNodeData = n1.getData(Galois::NONE);
      for (unsigned j=i+1; j<newNodes->size(); j++) {
	if (i != j) {
	  GNode n2 = (*newNodes)[j];;
	  Element& e = n2.getData(Galois::NONE);
	  
	  bool found = false;
	  int indexForNewNode = -1;
	  int indexForNode = -1;
	  
	  for (int x=2; x>=1; x--) {
	    for (int y=2; y>=1; y--) {
	      if (newNodeData.getPoint(x) == e.getPoint(y)) {
		indexForNewNode = x & 2, indexForNode = y & 2, found = true;
	      }
	    }
	  }
	  
	  if (found) {
	    graph->addEdge(n1, n2, indexForNewNode,	Galois::CHECK_CONFLICT);
	    graph->addEdge(n2, n1, indexForNode, Galois::CHECK_CONFLICT);
	  }
	}
      }
    }

    deletingNodes.insert(node);

    int size = newNodes->size();
    for (GNodeSet::iterator iter = deletingNodes.begin();
	 iter != deletingNodes.end(); ++iter) {
      GNode dnode = *iter;
      TupleList& tuples = dnode.getData(Galois::NONE).getTuples();

      for(TupleList::reverse_iterator list_iter = tuples.rbegin(), end = tuples.rend(); list_iter != end; ++list_iter) {
	  Tuple tup=*list_iter;
	  for (int i = 0; i < size; i++) {
	    Element& element = (*newNodes)[i].getData(Galois::NONE);
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

      Element& nodeData = dnode.getData(Galois::NONE);
      nodeData.getTuples().clear();
      graph->removeNode(dnode, Galois::NONE);
    }

    for (GNodeVector::iterator iter = newNodes->begin(); iter != newNodes->end(); )
      {
        if ((*iter).getData(Galois::NONE).getTuples().empty())
	  iter = newNodes->erase(iter);
        else
	  iter++;
      }
  }
};

#endif /* CAVITY_H_ */
