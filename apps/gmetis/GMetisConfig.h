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

#ifndef GMETISCONFIG_H_
#define GMETISCONFIG_H_

#include "MetisNode.h"
#include "ArraySet.h"

#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Accumulator.h"

#include <stdlib.h>

typedef int METISINT;
typedef double METISDOUBLE;


typedef Galois::Graph::FirstGraph<MetisNode,METISINT, true>            GGraph;
typedef Galois::Graph::FirstGraph<MetisNode,METISINT, true>::GraphNode GNode;

#include <set>
using namespace std;
typedef ArraySet< GNode > GNodeSet;
typedef set< GNode, std::less<GNode>, GaloisRuntime::MM::FSBGaloisAllocator<GNode> > GNodeSTLSet;
//typedef vector<GNode> GNodeSTLSet;

int getRandom(int num);

struct PerCPUValue {
  Galois::GAccumulator<int> mincutInc;
  Galois::GSetAccumulator<GNodeSTLSet> changedBndNodes;
};

struct gNodeToInt {
  GGraph* graph;
  gNodeToInt(GGraph* _g) :graph(_g) {}
  int operator()(GNode node){
    return graph->getData(node).getNodeId();
  }
};

int intlog2(int a);
#endif /* GMETISCONFIG_H_ */
