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



//#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/NonRemovingGraph.h"
#include "Galois/Galois.h"
#include "MetisNode.h"
#include "Galois/Timer.h"
#include "Galois/Runtime/mm/mem.h"
#include <stdlib.h>
#include "ArraySet.h"
typedef int METISINT;
typedef double METISDOUBLE;


typedef Galois::Graph::FirstGraph<MetisNode,METISINT, true>            GGraph;
typedef Galois::Graph::FirstGraph<MetisNode,METISINT, true>::GraphNode GNode;

#include <set>
using namespace std;
typedef ArraySet< GNode > GNodeSet;
struct GNodeSetCompare
{
  bool operator()(const GNode s1, const GNode s2) const
  {
    return s1.getData(Galois::NONE).getNodeId() < s2.getData(Galois::NONE).getNodeId();
  }
};
typedef set< GNode, GNodeSetCompare, GaloisRuntime::MM::FSBGaloisAllocator<GNode> > GNodeSTLSet;
//typedef vector<GNode> GNodeSTLSet;

template <typename T>
void arrayFill(T* array, int length, T value){
	for(int i=0;i<length;++i){
		array[i] = value;
	}
}

int gNodeToInt(GNode node);
int getRandom(int num);

struct PerCPUValue{
	int mincutInc;
	GNodeSTLSet changedBndNodes;
	PerCPUValue(){
		mincutInc = 0;
	}
};

struct mergeP {
  void operator()(PerCPUValue& a, PerCPUValue& b);
};

int intlog2(int a);
#endif /* GMETISCONFIG_H_ */
