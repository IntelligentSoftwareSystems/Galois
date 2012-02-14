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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef CAVITY_H
#define CAVITY_H

#include "Graph.h"

#include <vector>

//! A cavity which will be retrangulated
template<typename Alloc=std::allocator<char> >
class Cavity: private boost::noncopyable {
  typedef std::vector<GNode, Alloc> GNodeVector;
  typedef std::vector<std::pair<GNode,int>, Alloc> GNodeIntPairVector;

  struct InCircumcenter {
    Tuple tuple;
    InCircumcenter(const Tuple& t): tuple(t) { }
    bool operator()(const GNode& n) const {
      Element& e = n.getData(Galois::NONE);
      return e.inCircle(tuple);
    }
  };

  Searcher<Alloc> searcher;
  GNodeVector newNodes;
  GNode center;
  Point* point;
  Graph& graph;
  const Alloc& alloc;

  //! Find triangles that border cavity but are not in the cavity
  void findOutside(GNodeIntPairVector& outside) {
    for (typename Searcher<Alloc>::GNodeVector::iterator ii = searcher.inside.begin(),
        ei = searcher.inside.end(); ii != ei; ++ii) {

      for (Graph::neighbor_iterator jj = graph.neighbor_begin(*ii, Galois::NONE),
          ej = graph.neighbor_end(*ii, Galois::NONE); jj != ej; ++jj) {

        // i.e., if (!e.boundary() && e.inCircle(point->t())) 
        if (std::find(searcher.matches.begin(), searcher.matches.end(), *jj)
            != searcher.matches.end())
          continue;

        int index = graph.getEdgeData(*jj, *ii, Galois::NONE);
        outside.push_back(std::make_pair(*jj, index));

        Element& e = jj->getData(Galois::NONE);
        Point* p2 = e.getPoint(index);
        Point* p3 = e.getPoint((index + 1) % 3);

        p2->acquire(Galois::CHECK_CONFLICT);
        p3->acquire(Galois::CHECK_CONFLICT);
      }
    }
  }

  void addElements(GNodeIntPairVector& outside) {
    GNodeVector newNodes(alloc);

    // Create new nodes
    for (typename GNodeIntPairVector::iterator ii = outside.begin(),
        ei = outside.end(); ii != ei; ++ii) {
      const GNode& n = ii->first;
      int& index = ii->second;

      Element& e = n.getData(Galois::NONE);

      Point* p2 = e.getPoint(index);
      Point* p3 = e.getPoint((index + 1) % 3);

      Element newE(point, p2, p3);
      GNode newNode = graph.createNode(newE);

      point->addElement(newNode);
      p2->addElement(newNode);
      p3->addElement(newNode);

      graph.addNode(newNode, Galois::NONE);
      graph.addEdge(newNode, n, 1, Galois::NONE);
      graph.addEdge(n, newNode, index, Galois::NONE);
      
      newNodes.push_back(newNode);
    }

    // Update new node connectivity
    for (unsigned i = 0; i < newNodes.size(); ++i) {
      const GNode& n1 = newNodes[i];
      const Element& e1 = n1.getData(Galois::NONE);
      for (unsigned j = i + 1; j < newNodes.size(); ++j) {
	if (i != j) {
	  const GNode& n2 = newNodes[j];
	  const Element& e2 = n2.getData(Galois::NONE);
	  
	  bool found = false;
	  int indexForNewNode;
	  int indexForNode;
	  
	  for (int x = 2; x >= 1; --x) {
	    for (int y = 2; y >= 1; --y) {
	      if (e1.getPoint(x) == e2.getPoint(y)) {
		indexForNewNode = x & 2;
                indexForNode = y & 2;
                found = true;
	      }
	    }
	  }
	  
	  if (found) {
	    graph.addEdge(n1, n2, indexForNewNode, Galois::NONE);
	    graph.addEdge(n2, n1, indexForNode, Galois::NONE);
	  }
	}
      }
    }
  }

  void removeElements() {
    for (typename Searcher<Alloc>::GNodeVector::iterator ii = searcher.matches.begin(),
        ei = searcher.matches.end(); ii != ei; ++ii) {
      graph.removeNode(*ii, Galois::NONE);
    }
  }

public:
  Cavity(Graph& g, const Alloc& a = Alloc()):
    searcher(g, a),
    newNodes(a),
    graph(g),
    alloc(a)
    { }

  void init(const GNode& c, Point* p) {
    searcher.useMark(p, 1, p->numTries());
    center = c;
    point = p;
  }

  void build() {
    assert(center.getData().inCircle(point->t()));
    searcher.findAll(center, InCircumcenter(point->t()));
    assert(!searcher.inside.empty());
  }

  void update() {
    GNodeIntPairVector outside(alloc);
    findOutside(outside);
    removeElements();
    addElements(outside);
  }
};

#endif
