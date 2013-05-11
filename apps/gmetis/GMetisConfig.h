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

#include "Galois/Graph/Graph.h"
#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Accumulator.h"
#include "Galois/Graph/memScalGraph.h"
#include <stdlib.h>
//#include "Galois/Graph/LCGraph.h"
#include "Galois/Graph/LC_Morph_Graph.h"

typedef int METISINT;
typedef double METISDOUBLE;


/*

typedef Galois::Graph::FirstGraph<MetisNode,METISINT, true>            GGraph;
typedef Galois::Graph::FirstGraph<MetisNode,METISINT, true>::GraphNode GNode;

*/
#define localNodeData
#define LC_MORPH
#ifndef LC_MORPH

typedef Galois::Graph::MemScalGraph<MetisNode,METISINT, true>            GGraph;
typedef Galois::Graph::MemScalGraph<MetisNode,METISINT, true>::GraphNode GNode;
#else
/*
 *I imagine this might become a graph where instead of int being the edge we might be storing a struct of int and pair<int,int> 
 *The first int will be the normal weight, second would be a pair specifying partition number and weight related to that partition.
 */
typedef Galois::Graph::LC_Morph_Graph<MetisNode,METISINT> GGraph;
typedef Galois::Graph::LC_Morph_Graph<MetisNode,METISINT>::GraphNode GNode;
#endif

#include <set>
using namespace std;

namespace testMetis {
	extern bool testCoarsening;
	extern bool testInitialPartition;
}

namespace variantMetis {
	extern bool mergeMatching;
	extern bool bagRefining;
	extern bool noPartInfo;
}

typedef ArraySet< GNode > GNodeSet;

typedef set< GNode, std::less<GNode>, Galois::Runtime::MM::FSBGaloisAllocator<GNode> > GNodeSTLSet;
//typedef vector<GNode> GNodeSTLSet;
typedef Galois::Runtime::PerThreadVector<GNode> NodeCtxWl;

int getRandom(int num);

struct PerCPUValue {
  Galois::GAccumulator<int> mincutInc;
  Galois::GSetAccumulator<GNodeSTLSet> changedBndNodes;
};

struct IteratorPairs{
	GGraph::edge_iterator first_start;
	GGraph::edge_iterator first_end;
	GGraph::edge_iterator second_start;
	GGraph::edge_iterator second_end;
	GNode node;
	typedef GGraph::edge_iterator ei;
	IteratorPairs(ei first_start,ei first_end,ei second_start,ei second_end,GNode node) {
		this->first_start = first_start;
		this->first_end = first_end;
		this->second_start = second_start;
		this->second_end = second_end;
		this->node = node;
	}
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
