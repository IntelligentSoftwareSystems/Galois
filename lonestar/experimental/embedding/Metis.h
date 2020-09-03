/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef METIS_H_
#define METIS_H_

#include "galois/graphs/MorphHyperGraph.h"
#include "galois/AtomicWrapper.h"

class MetisNode;
using GGraph   = galois::graphs::MorphHyperGraph<MetisNode, int, true>;
using GNode    = GGraph::GraphNode;
using GNodeBag = galois::InsertBag<GNode>;

// algorithms
enum InitialPartMode {GGP, GGGP, MGGGP };
enum refinementMode {FM, BKL2, ROBO, GRACLUS};
enum coarseMode {HMETIS, PAIR, BOTH};
enum pairScheduleMode {FIRST, MAXW, EC};
enum scheduleMode {PLD, WD, RI, PP, MRI, MWD, DEG, MDEG, HIS};

enum coarseModeII {HMETISII, PAIRII};
enum pairScheduleModeII {FIRSTII, MAXWII, ECII};
enum scheduleModeII {PLDII, WDII, RIII, PPII, MRIII, MWDII, DEGII, MDEGII, HISII};
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
    data.cd.parent      = NULL;
    netval = 0;
  }

public:
  bool flag;
  GNode minnode;
  int dim;
  int emb;
  galois::CopyableAtomic<float> gain;
  unsigned counter;
  float icnt;
  int gains;
  unsigned p1;
  unsigned p2;
  galois::CopyableAtomic<float> sum2;
  galois::CopyableAtomic<float> sum1;
  int nodeid;
  galois::CopyableAtomic<int> cnt;
  galois::CopyableAtomic<int> FS;
  galois::CopyableAtomic<int> TE;
  //int FS[14];
  //int TE[14];
 // std::array<galois::CopyableAtomic<int>, 10>  FS1;
 // std::array<galois::CopyableAtomic<int>, 10> TE1;
  galois::CopyableAtomic<int> netnum;
  galois::CopyableAtomic<double> netval;
  void initPartition() { 
    pd.locked = false;
    gain = 0.0;
    gains = 0;
  }

  // int num;
  explicit MetisNode(int weight) : _weight(weight) {
    initCoarsen();
    initPartition();
    counter = 0;
    data.rd.partition = 0;
  }

  MetisNode(unsigned weight, GNode child0, GNode child1 = NULL)
      : _weight(weight) {
    initCoarsen();
    initPartition();
    children[0] = child0;
    children[1] = child1;
    counter = 0;
    data.rd.partition = 0;
  }

  MetisNode() : _weight(1) {
    initCoarsen();
    initPartition();
    counter = 0;
    data.rd.partition = 0;
    flag = false;
    p1 = 0;
    p2 = 0;
  }

  // call to switch data to refining
  void initRefine(unsigned part = 0, bool bound = false) {
    refineData rd = {part, part, bound};
    data.rd       = rd;
    counter = 0;
  }

  int getWeight() const { return _weight; }
  void setWeight(int weight) { _weight = weight; }


  void setParent(GNode p) { data.cd.parent = p; }
  GNode getParent() const {
    assert(data.cd.parent);
    return data.cd.parent;
  }
  int getGain() {
    return FS - (TE + counter);
    //return gains - counter;
  }
  int getGains() {
    return FS - (TE + counter);
    //return gains - counter;
  }

  void setMatched() { data.cd.matched = true; }
  void notMatched() { data.cd.matched = false; }
  bool isMatched() const { return data.cd.matched; }

  void setFailedMatch() { data.cd.failedmatch = true; }
  bool isFailedMatch() const { return data.cd.failedmatch; }

  GNode getChild(unsigned x) const { return children[x]; }
  void setChild(GNode c) {children.push_back(c);}
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

  unsigned getNumNodes() { return std::distance(graph.cellList().begin(), graph.cellList().end()); }

  unsigned getTotalWeight() {
    MetisGraph* f = this;
    while (f->finer)
      f = f->finer;
    return std::distance(f->graph.cellList().begin(), f->graph.cellList().end());
  }
};

// Structure to store working partition information
struct partInfo {
  unsigned partNum;
  unsigned partMask;
  unsigned partWeight;
  unsigned iter;
  std::vector<GNode> gainlist;
  explicit partInfo(unsigned mw) : partNum(0), partMask(1), partWeight(mw) {iter = 0;}

  partInfo() : partNum(~0), partMask(~0), partWeight(~0) {iter = 0;}

  partInfo(unsigned pn, unsigned pm, unsigned pw)
      : partNum(pn), partMask(pm), partWeight(pw) {iter = 0;}

  unsigned splitID() const { return partNum | partMask; }
  void setgain(GNode n) { gainlist.push_back(n);}
  GNode getgain() {
    return gainlist[iter++];
  }
  int size () {return gainlist.size();}
  int getSize() { return gainlist.size() - iter; }

  std::pair<unsigned, unsigned> splitRatio(unsigned numParts) {
    unsigned L = 0, R = 0;
    unsigned LM = partMask - 1; // 00100 -> 00011
    for (unsigned x = 0; x < numParts; ++x)
      if ((x & LM) == partNum) {
        if (x & partMask)
          ++R;
        else
          ++L;
      }
    return std::make_pair(L, R);
  }

  partInfo split() {
    partInfo np(splitID(), partMask << 1, 0);
    partMask <<= 1;
    return np;
  }
};

std::ostream& operator<<(std::ostream& os, const partInfo& p);

// Metrics
void printPartStats(std::vector<partInfo>&);
unsigned graphStat(GGraph& graph);
std::vector<unsigned> edgeCut(GGraph& g, unsigned nparts);
void printCuts(const char* str, MetisGraph* g, unsigned numPartitions);
unsigned computeCut(GGraph& g);

// Coarsening
MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo,
                    bool verbose, scheduleMode sMode, scheduleModeII sModeII);

// Partitioning
void partition(MetisGraph* coarseMetisGraph);
std::vector<partInfo> BisectAll(MetisGraph* mcg, unsigned numPartitions,
                                unsigned maxSize);
// Refinement
//void refine(MetisGraph* coarseGraph, std::vector<partInfo>& parts,
//             refinementMode refM, bool verbose);
 void refine(MetisGraph* coarseGraph, unsigned refineTo);
// Balancing
void balance(MetisGraph* Graph, std::vector<partInfo>& parts, unsigned maxSize);
void normalize(MetisGraph* mcg, std::string filename);

#endif
