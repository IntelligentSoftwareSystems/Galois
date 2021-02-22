/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef BIPART_H_
#define BIPART_H_

#include "galois/graphs/LC_CSR_Graph.h"
#include "galois/AtomicWrapper.h"

class MetisNode;
typedef uint32_t EdgeTy;

struct GGraph
    : public galois::graphs::LC_CSR_Graph<MetisNode, EdgeTy>::with_no_lockable<
        true>::type::with_numa_alloc<true>::type {
    	//false>::type::with_numa_alloc<true>::type {
  size_t hedges;
  size_t hnodes;
};

using GNode    = GGraph::GraphNode;
using GNodeBag = galois::InsertBag<GNode>;

constexpr galois::MethodFlag flag_no_lock = galois::MethodFlag::UNPROTECTED;
// algorithms
enum scheduleMode { PLD, WD, RI, PP, MRI, MWD, DEG, MDEG, HIS, RAND };

enum coarseModeII { HMETISII, PAIRII };
enum pairScheduleModeII { FIRSTII, MAXWII, ECII };
// Nodes in the metis graph
class MetisNode {

  struct coarsenData {
    int matched : 1;
    int failedmatch : 1;
    GNode parent;
  };
  struct refineData {
    unsigned partition;
    unsigned oldPartition;
    bool maybeBoundary;
  };
  struct partitionData {
    bool locked;
  };

  partitionData pd;

  void initCoarsen() {
    data.cd.matched     = false;
    data.cd.failedmatch = false;
    data.cd.parent      = 0;
    netval              = 0;
  }

public:
  // bool flag;
  unsigned counter;
  int nodeid;
  galois::CopyableAtomic<int> FS;
  galois::CopyableAtomic<int> TE;
  galois::CopyableAtomic<int> netnum;
  galois::CopyableAtomic<int> netrand;
  galois::CopyableAtomic<int> netval;
  galois::CopyableAtomic<int> degree;
  /*std::atomic<int> FS;
	std::atomic<int> TE;
	std::atomic<int> netnum;
	std::atomic<int> netrand;
	std::atomic<int> netval;
	std::atomic<int> degree;
*/	uint32_t index;
	bool notAlone;
	
	void initPartition() { pd.locked = false; }
	
	
  // int num;
  explicit MetisNode(int weight) : _weight(weight) {
    initCoarsen();
    initPartition();
    counter           = 0;
    data.rd.partition = 0;
  }

  MetisNode(unsigned weight, GNode child0, GNode child1 = 0) : _weight(weight) {
    initCoarsen();
    initPartition();
    children[0]       = child0;
    children[1]       = child1;
    counter           = 0;
    data.rd.partition = 0;
  }

  MetisNode() : _weight(1) {
    initCoarsen();
    initPartition();
    counter           = 0;
    data.rd.partition = 0;
    data.cd.matched   = false;
  }

  // call to switch data to refining
  void initRefine(unsigned part = 0, bool bound = false) {
    refineData rd = {part, part, bound};
    data.rd       = rd;
    counter       = 0;
  }

  int getWeight() const { return _weight; }
  void setWeight(int weight) { _weight = weight; }

  void setParent(GNode p) { data.cd.parent = p; }
  GNode getParent() const {
    assert(data.cd.parent);
    return data.cd.parent;
  }
  int getGain() { return FS - (TE + counter); }

  void setMatched() { data.cd.matched = true; }
  void notMatched() { data.cd.matched = false; }
  bool isMatched() const { return data.cd.matched; }

  void setFailedMatch() { data.cd.failedmatch = true; }
  bool isFailedMatch() const { return data.cd.failedmatch; }

  GNode getChild(unsigned x) const { return children[x]; }
  void setChild(GNode c) { children.push_back(c); }
  unsigned numChildren() const { return children.size(); }

  unsigned getPart() const { return data.rd.partition; }
  void setPart(unsigned val) { data.rd.partition = val; }

  int getOldPart() const { return data.rd.oldPartition; }
  void OldPartCpyNew() { data.rd.oldPartition = data.rd.partition; }

  bool getmaybeBoundary() const { return data.rd.maybeBoundary; }
  void setmaybeBoundary(bool val) { data.rd.maybeBoundary = val; }

  void setLocked(bool locked) { pd.locked = locked; }
  bool isLocked() { return pd.locked; }

private:
  union {
    coarsenData cd;
    refineData rd;
  } data;

  std::vector<GNode> children;
  unsigned _weight;
};

// Structure to keep track of graph hirarchy
class MetisGraph {
  MetisGraph* coarser;
  MetisGraph* finer;

  GGraph graph;

public:
  MetisGraph() : coarser(0), finer(0) {}

  explicit MetisGraph(MetisGraph* finerGraph) : coarser(0), finer(finerGraph) {
    finer->coarser = this;
  }

  const GGraph* getGraph() const { return &graph; }
  GGraph* getGraph() { return &graph; }
  MetisGraph* getFinerGraph() const { return finer; }
  MetisGraph* getCoarserGraph() const { return coarser; }

  // unsigned getNumNodes() { return std::distance(graph.cellList().begin(),
  // graph.cellList().end()); }

  unsigned getTotalWeight() {
    MetisGraph* f = this;
    while (f->finer)
      f = f->finer;
    // return std::distance(f->graph.cellList().begin(),
    // f->graph.cellList().end());
    return 0;
  }
};

// Metrics
unsigned graphStat(GGraph& graph);
// Coarsening
MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo,
                    scheduleMode sMode);

// Partitioning
void partition(MetisGraph*, unsigned);
// Refinement
void refine(MetisGraph* coarseGraph, unsigned K, double imbalance);

#endif
