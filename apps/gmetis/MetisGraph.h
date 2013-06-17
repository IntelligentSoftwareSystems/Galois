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

#ifndef METISGRAPH_H_
#define METISGRAPH_H_
#include "MetisNode.h"
#include "GMetisConfig.h"
#include <vector>
#include <iostream>
#include "Galois/Atomic.h"
using namespace std;

class MetisGraph{
  typedef Galois::GAtomic<int> AtomicInteger;

  MetisGraph* coarser;
  MetisGraph* finer;

  GGraph graph;

public:
  MetisGraph()
    :coarser(nullptr), finer(nullptr)
  {}
  
  explicit MetisGraph(MetisGraph* finerGraph)
    :coarser(0), finer(finerGraph) {
    finer->coarser = this;
  }
  
  const GGraph* getGraph() const { return &graph; }
  GGraph* getGraph() { return &graph; }
  MetisGraph* getFinerGraph() const { return finer; }
  MetisGraph* getCoarserGraph() const { return coarser; }

  unsigned getNumNodes() {
    return std::distance(graph.begin(), graph.end());
  }
  
  unsigned getTotalWeight() {
    MetisGraph* f = this;
    while (f->finer)
      f = f->finer;
    return std::distance(f->graph.begin(), f->graph.end());
  }

};
#endif /* METISGRAPH_H_ */
