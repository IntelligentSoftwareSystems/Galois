#ifndef METIS_H_
#define METIS_H_

#include "galois/graphs/LC_Morph_Graph.h"

class MetisNode;
using GGraph = galois::graphs::LC_Morph_Graph<MetisNode, int>;
using GNode = GGraph::GraphNode;
using GNodeBag = galois::InsertBag<GNode>;


//algorithms
enum InitialPartMode {GGP, GGGP, MGGGP};
enum refinementMode {BKL, BKL2, ROBO, GRACLUS};
//Nodes in the metis graph
class MetisNode {

  struct coarsenData {
    int matched:1;
    int failedmatch:1;
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

  void initCoarsen(){
    data.cd.matched = false;
    data.cd.failedmatch = false;
    data.cd.parent = NULL;
  }

public:
  void initPartition() {
    pd.locked = false;
  }
  //int num;
  explicit MetisNode(int weight) :_weight(weight) {
    initCoarsen();
    initPartition();
  }

  MetisNode(unsigned weight, GNode child0, GNode child1 = NULL)
    : _weight(weight) {
    initCoarsen();
    initPartition();
    children[0] = child0;
    children[1] = child1;
  }

  MetisNode():_weight(1) { initCoarsen(); initPartition();}

  //call to switch data to refining
  void initRefine(unsigned part = 0, bool bound = false) {
    refineData rd = {part, part, bound};
    data.rd = rd;
  }

  int getWeight() const { return _weight; }
  void setWeight(int weight) { _weight = weight; }

  void setParent(GNode p)  { data.cd.parent = p; }
  GNode getParent() const  {  assert(data.cd.parent); return data.cd.parent; }

  void setMatched() { data.cd.matched = true; }
  bool isMatched() const   { return data.cd.matched; }

  void setFailedMatch() { data.cd.failedmatch = true; }
  bool isFailedMatch() const { return data.cd.failedmatch; }

  GNode getChild(unsigned x) const { return children[x]; }
  unsigned numChildren() const { return children[1] ? 2 : 1; }

  unsigned getPart() const { return data.rd.partition; }
  void setPart(unsigned val) { data.rd.partition = val; }

  int getOldPart() const {return data.rd.oldPartition;}
  void OldPartCpyNew(){ data.rd.oldPartition = data.rd.partition; }

  bool getmaybeBoundary() const {return data.rd.maybeBoundary; }
  void setmaybeBoundary(bool val){ data.rd.maybeBoundary = val; }

  void setLocked(bool locked) { pd.locked = locked;}
  bool isLocked() { return pd.locked;}

private:
  union {
    coarsenData cd;
    refineData rd;
  } data;

  GNode children[2];
  unsigned _weight;
};

//Structure to keep track of graph hirarchy
class MetisGraph{
  MetisGraph* coarser;
  MetisGraph* finer;

  GGraph graph;

public:
  MetisGraph() :coarser(0), finer(0) { }

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


//Structure to store working partition information
struct partInfo {
  unsigned partNum;
  unsigned partMask;
  unsigned partWeight;

  explicit partInfo(unsigned mw)
    :partNum(0), partMask(1), partWeight(mw) {}

  partInfo() :partNum(~0), partMask(~0), partWeight(~0) {}

  partInfo(unsigned pn, unsigned pm, unsigned pw) :partNum(pn), partMask(pm), partWeight(pw) {}

  unsigned splitID() const {
    return partNum | partMask;
  }

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

//Metrics
void printPartStats(std::vector<partInfo>&);
unsigned graphStat(GGraph& graph);
std::vector<unsigned> edgeCut(GGraph& g, unsigned nparts);
void printCuts(const char* str, MetisGraph* g, unsigned numPartitions);
unsigned computeCut(GGraph& g);

//Coarsening
MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo, bool verbose);

//Partitioning
std::vector<partInfo> partition(MetisGraph* coarseMetisGraph, unsigned fineMetisGraphWeight, unsigned numPartitions, InitialPartMode partMode);
std::vector<partInfo> BisectAll(MetisGraph* mcg, unsigned numPartitions, unsigned maxSize);
//Refinement
void refine(MetisGraph* coarseGraph, std::vector<partInfo>& parts, unsigned minSize, unsigned maxSize, refinementMode refM, bool verbose);
//void refinePart(GGraph& g, std::vector<partInfo>& parts, unsigned maxSize);
//Balancing
void balance(MetisGraph* Graph, std::vector<partInfo>& parts, unsigned maxSize);

#endif
