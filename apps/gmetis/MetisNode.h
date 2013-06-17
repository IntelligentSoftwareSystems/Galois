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

#ifndef METISNODE_H_
#define METISNODE_H_

typedef int METISINT;
typedef double METISDOUBLE;
#include <stddef.h>
#include <vector>
#include "llvm/ADT/SmallVector.h"
#include <iostream>
#include "Galois/gdeque.h"
#include "Galois/Graph/LC_Morph_Graph.h"


class MetisNode;
typedef Galois::Graph::LC_Morph_Graph<MetisNode,METISINT> GGraph;
typedef Galois::Graph::LC_Morph_Graph<MetisNode,METISINT>::GraphNode GNode;

using namespace std;
class MetisNode{

public:
  explicit MetisNode(int weight) :_weight(weight) {
    init();
  }
  
  MetisNode(GNode child0, unsigned weight)
    : _weight(weight) {
    children[0] = child0;
    init();
    onlyOneChild = true;
  }

  MetisNode(GNode child0, GNode child1, unsigned weight)
    : onlyOneChild(false), _weight(weight) {
    children[0] = child0;
    children[1] = child1;
    init();
  }

  MetisNode() = default;

  void init(){
    _numEdges = 0;
    _weightEdge = 0;
    _partition = 0;
    bmatched = false;
    bparent = false;
    onlyOneChild = false;
  }

  int getWeight() const { return _weight; }
  void setWeight(int weight) { _weight = weight; }
  
  unsigned getEdgeWeight() const { return _weightEdge; }
  void setEdgeWeight(unsigned w) { _weightEdge = w; }

  //ADL
  void setParent(GNode p)  { parent = p; bparent = true; }
  GNode getParent() const  { assert(bparent); return parent; }

  void setMatched(GNode v) { matched = v; bmatched = true; }
  GNode getMatched() const { assert(bmatched); return matched; }
  bool isMatched() const   { return bmatched; }

  GNode getChild(unsigned x) const { return children[x]; }
  unsigned numChildren() const { return onlyOneChild ? 1 : 2; }

  unsigned getNumEdges() const { return _numEdges; }
  void setNumEdges(unsigned val) { _numEdges = val; }

  unsigned getPart() const { return _partition; }
  void setPart(unsigned val) { _partition = val; }

private:
  bool bmatched;
  GNode matched;
  bool bparent;
  GNode parent;
  GNode children[2];
  bool onlyOneChild;
  unsigned _weight;
  unsigned _numEdges;
  unsigned _partition;
  unsigned _weightEdge;
};

#endif /* METISNODE_H_ */
