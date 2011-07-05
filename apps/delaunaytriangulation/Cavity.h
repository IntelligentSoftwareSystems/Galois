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

  GNodeDeque oldNodes;
  GNodeDeque connectionNodes;
  GNodeSet deletingNodes;
  GNode node;
  Tuple tuple;
  Galois::PerIterAllocTy& _cnx;

public:
  typedef std::vector<GNode, PerIterGNodeAlloc> GNodeVector;
  
 Cavity(Graph* g, GNode& n, Tuple& t, Galois::PerIterAllocTy& cnx):
   graph(g),
   oldNodes(cnx),
   connectionNodes(cnx),
   deletingNodes(std::less<GNode>(), cnx),
   node(n),
   tuple(t),
   _cnx(cnx)
  {}

  void build() {
    GNodeVector frontier(_cnx);
    //std::vector<GNode> frontier;
    frontier.push_back(node);
    while (!frontier.empty()){
      GNode curr = frontier.back();
      frontier.pop_back();
      for (Graph::neighbor_iterator ii = graph->neighbor_begin(curr, Galois::Graph::CHECK_CONFLICT),
          ee = graph->neighbor_end(curr, Galois::Graph::CHECK_CONFLICT);
          ii != ee; ++ii) {
        GNode neighbor = *ii;
        Element& neighborElement = neighbor.getData(Galois::Graph::CHECK_CONFLICT);

        if (!graph->containsNode(neighbor) || neighbor == node || deletingNodes.find(neighbor) != deletingNodes.end()) {
          continue;
        }
        if (neighborElement.getBDim() && neighborElement.inCircle(tuple)) {
          deletingNodes.insert(neighbor);
          frontier.push_back(neighbor);
        } else {
          oldNodes.push_back(curr);
          connectionNodes.push_back(neighbor);
        }
      }
    }
  }

  void update(GNodeVector* newNodes){
    Element& nodeData = node.getData(Galois::Graph::NONE);
    nodeData.getTuples().pop_back();
    //vector<Element, Galois::PerIterMem::ItAllocTy::rebind<Element>::other> newElements;
    //vector<Element*> newElements;
    while (!connectionNodes.empty()) {
      GNode neighbor = connectionNodes.front();
      connectionNodes.pop_front();
      GNode oldNode = oldNodes.front();
      oldNodes.pop_front();
      int index = graph->getEdgeData(neighbor, oldNode, Galois::Graph::NONE);
      Element& neighborElement = neighbor.getData(Galois::Graph::NONE);
      Element e(tuple, neighborElement.getPoint(index), neighborElement.getPoint((index + 1) % 3));
      GNode nnode = graph->createNode(e);
      graph->addNode(nnode, Galois::Graph::NONE);
      graph->addEdge(nnode, neighbor, 1, Galois::Graph::NONE);
      graph->addEdge(neighbor, nnode, index, Galois::Graph::NONE);

      int numNeighborsFound = 0;
      GNodeVector::iterator nEnd = newNodes->end();
      for (GNodeVector::iterator iter=newNodes->begin();iter!=nEnd;iter++) {
        GNode newNode = *iter;
        bool found = false;
        int indexForNewNode = -1;
        int indexForNode = -1;
        Element& newNodeData = newNode.getData(Galois::Graph::NONE);

        for (int x=2; x>=1; x--) {
          for (int y=2; y>=1; y--) {
            if (newNodeData.getPoint(x) == e.getPoint(y)) {
              indexForNewNode = x & 2, indexForNode = y & 2, found = true;
            }
          }
        }

        if (found) {
          graph->addEdge(newNode, nnode, indexForNewNode,
              Galois::Graph::CHECK_CONFLICT);
          graph->addEdge(nnode, newNode, indexForNode, 
              Galois::Graph::CHECK_CONFLICT);
          numNeighborsFound++;
        }
        if (numNeighborsFound == 2) {
          break;
        }
      }

      newNodes->push_back(nnode);
      Element& nnode_data = nnode.getData(Galois::Graph::NONE);
      //newElements.push_back(&nnode_data);

      Element& oldNodeData = oldNode.getData(Galois::Graph::NONE);
      TupleList& ntuples = nnode_data.getTuples();
      TupleList& tuples = oldNodeData.getTuples();
      if (!tuples.empty()) {
        TupleList newTuples;
        unsigned tSize = tuples.size();
        for (unsigned i=0; i<tSize; i++) {
          Tuple t=tuples[i];
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

    deletingNodes.insert(node);
    dispatchTuples(newNodes);
    for (GNodeSet::iterator setIter = deletingNodes.begin();
        setIter != deletingNodes.end(); ++setIter) {
      GNode dnode = *setIter;
      graph->removeNode(dnode, Galois::Graph::NONE);
    }
  }

  void dispatchTuples(GNodeVector* newNodes) {
    int size = newNodes->size();
    for (GNodeSet::iterator iter = deletingNodes.begin();
        iter != deletingNodes.end(); ++iter) {
      GNode dnode = *iter;
      TupleList& tuples = dnode.getData(Galois::Graph::NONE).getTuples();

      unsigned tSize = tuples.size();
      for (unsigned j=0; j<tSize; j++) {
        Tuple tup=tuples[tSize-j-1];

        for (int i = 0; i < size; i++) {
          Element& element = (*newNodes)[i].getData(Galois::Graph::NONE);
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
