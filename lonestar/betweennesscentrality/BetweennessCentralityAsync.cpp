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

#include "Lonestar/BoilerPlate.h"

#include "galois/graphs/BufferedGraph.h"
#include "galois/graphs/B_LC_CSR_Graph.h"
#include "galois/Bag.h"

#include "BCNode.h"
#include "BCEdge.h"

#include <iomanip>

// WARNING: optimal chunk size may differ depending on input graph
constexpr static const unsigned CHUNK_SIZE = 64u;

////////////////////////////////////////////////////////////////////////////////
// Command line parameters
////////////////////////////////////////////////////////////////////////////////

namespace cll = llvm::cl;

static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input graph in Galois bin "
                                                "format>"),
                                      cll::Required);

static cll::opt<std::string> sourcesToUse("sourcesToUse",
                                          cll::desc("Whitespace separated list "
                                                    "of sources in a file to "
                                                    "use in BC"),
                                          cll::init(""));

// @todo fix how this works
//static cll::opt<unsigned int> startNode("startNode",
//                                        cll::desc("Node to start search from"),
//                                        cll::init(0));

static cll::opt<unsigned int>
    numOfSources("numOfSources",
                 cll::desc("Number of sources to compute"
                           " BC on"),
                 cll::init(0));

static cll::opt<unsigned int>
    numOfOutSources("numOfOutSources",
                    cll::desc("Number of sources WITH EDGES "
                              " to compute BC on"),
                    cll::init(0));

static cll::opt<bool> generateCert("generateCertificate",
                                   cll::desc("Prints certificate at end of "
                                             "execution"),
                                   cll::init(false));
// TODO bring this back
// static cll::opt<bool> useNodeBased("useNodeBased",
//                                   cll::desc("Use node based execution"),
//                                   cll::init(true));

using NodeType = BCNode<BC_USE_MARKING, BC_CONCURRENT>;
using Graph    = galois::graphs::B_LC_CSR_Graph<NodeType, BCEdge, false, true>;

// Work items for the forward phase
struct ForwardPhaseWorkItem {
  uint32_t nodeID;
  uint32_t distance;
  ForwardPhaseWorkItem() : nodeID(infinity), distance(infinity){};
  ForwardPhaseWorkItem(uint32_t _n, uint32_t _d) : nodeID(_n), distance(_d){};
};

// grabs distance from a forward phase work item
struct FPWorkItemIndexer {
  uint32_t operator()(const ForwardPhaseWorkItem& it) const {
    return it.distance;
  }
};

// obim worklist type declaration
namespace gwl = galois::worklists;
using PSchunk = gwl::PerSocketChunkFIFO<CHUNK_SIZE>;
using OBIM    = gwl::OrderedByIntegerMetric<FPWorkItemIndexer, PSchunk>;

template <typename T, bool enable>
struct Counter: public T {
  std::string name;

  Counter(std::string s): name(std::move(s)) { }

  ~Counter() {
    galois::runtime::reportStat_Single("(NULL)", name, this->reduce());
  }
};

template <typename T>
struct Counter<T, false> {
  Counter(std::string s) { }

  template <typename... Args>
  void update(Args... args) { }
};

struct BetweenessCentralityAsync {
  Graph& graph;

  BetweenessCentralityAsync(Graph& _graph) : graph(_graph) {}

  using SumCounter = Counter<galois::GAccumulator<unsigned long>, BC_COUNT_ACTIONS>;
  SumCounter spfuCount{"SP&FU"};
  SumCounter updateSigmaP1Count{"UpdateSigmaBefore"};
  SumCounter updateSigmaP2Count{"RealUS"};
  SumCounter firstUpdateCount{"First Update"};
  SumCounter correctNodeP1Count{"CorrectNodeBefore"};
  SumCounter correctNodeP2Count{"Real CN"};
  SumCounter noActionCount{"NoAction"};

  using MaxCounter = Counter<galois::GReduceMax<unsigned long>,
                                            BC_COUNT_ACTIONS>;
  MaxCounter largestNodeDist{"Largest node distance"};

  using LeafCounter = Counter<galois::GAccumulator<unsigned long>, BC_COUNT_LEAVES>;

  void correctNode(uint32_t dstID, BCEdge& ed) {
    NodeType& dstData = graph.getData(dstID);

    // loop through in edges
    for (auto e : graph.in_edges(dstID)) {
      BCEdge& inEdgeData = graph.getInEdgeData(e);

      uint32_t srcID = graph.getInEdgeDst(e);
      if (srcID == dstID)
        continue;

      NodeType& srcData = graph.getData(srcID);

      // lock in right order
      if (srcID < dstID) {
        srcData.lock();
        dstData.lock();
      } else {
        dstData.lock();
        srcData.lock();
      }

      const unsigned edgeLevel = inEdgeData.level;

      // Correct Node
      if (srcData.distance >= dstData.distance) {
        correctNodeP1Count.update(1);
        dstData.unlock();

        if (edgeLevel != infinity) {
          inEdgeData.level = infinity;
          if (edgeLevel == srcData.distance) {
            correctNodeP2Count.update(1);
            srcData.nsuccs--;
          }
        }
        srcData.unlock();
      } else {
        srcData.unlock();
        dstData.unlock();
      }
    }
  }

  template <typename CTXType>
  void spAndFU(uint32_t srcID, uint32_t dstID, BCEdge& ed, CTXType& ctx) {
    spfuCount.update(1);

    NodeType& srcData = graph.getData(srcID);
    NodeType& dstData = graph.getData(dstID);

    // make dst a successor of src, src predecessor of dst
    srcData.nsuccs++;
    const ShortPathType srcSigma = srcData.sigma;
    assert(srcSigma > 0);
    NodeType::predTY& dstPreds = dstData.preds;
    bool dstPredsNotEmpty      = !dstPreds.empty();
    dstPreds.clear();
    dstPreds.push_back(srcID);
    dstData.distance = srcData.distance + 1;

    largestNodeDist.update(dstData.distance);

    dstData.nsuccs = 0;        // SP
    dstData.sigma  = srcSigma; // FU
    ed.val         = srcSigma;
    ed.level       = srcData.distance;
    srcData.unlock();
    if (!dstData.isAlreadyIn())
      ctx.push(ForwardPhaseWorkItem(dstID, dstData.distance));
    dstData.unlock();
    if (dstPredsNotEmpty) {
      correctNode(dstID, ed);
    }
  }

  template <typename CTXType>
  void updateSigma(uint32_t srcID, uint32_t dstID, BCEdge& ed, CTXType& ctx) {
    updateSigmaP1Count.update(1);

    NodeType& srcData = graph.getData(srcID);
    NodeType& dstData = graph.getData(dstID);

    const ShortPathType srcSigma = srcData.sigma;
    const ShortPathType eval     = ed.val;
    const ShortPathType diff     = srcSigma - eval;

    srcData.unlock();
    // greater than 0.0001 instead of 0 due to floating point imprecision
    if (diff > 0.0001) {
      updateSigmaP2Count.update(1);
      ed.val = srcSigma;

      // ShortPathType old = dstData.sigma;
      dstData.sigma += diff;

      // if (old >= dstData.sigma) {
      //  galois::gDebug("Overflow detected; capping at max uint64_t");
      //  dstData.sigma = std::numeric_limits<uint64_t>::max();
      //}

      int nbsuccs = dstData.nsuccs;

      if (nbsuccs > 0) {
        if (!dstData.isAlreadyIn())
          ctx.push(ForwardPhaseWorkItem(dstID, dstData.distance));
      }
      dstData.unlock();
    } else {
      dstData.unlock();
    }
  }

  template <typename CTXType>
  void firstUpdate(uint32_t srcID, uint32_t dstID, BCEdge& ed, CTXType& ctx) {
    firstUpdateCount.update(1);

    NodeType& srcData = graph.getData(srcID);
    srcData.nsuccs++;
    const ShortPathType srcSigma = srcData.sigma;

    NodeType& dstData = graph.getData(dstID);
    dstData.preds.push_back(srcID);

    const ShortPathType dstSigma = dstData.sigma;

    // ShortPathType old = dstData.sigma;
    dstData.sigma = dstSigma + srcSigma;
    // if (old >= dstData.sigma) {
    //  galois::gDebug("Overflow detected; capping at max uint64_t");
    //  dstData.sigma = std::numeric_limits<uint64_t>::max();
    //}

    ed.val   = srcSigma;
    ed.level = srcData.distance;
    srcData.unlock();
    int nbsuccs = dstData.nsuccs;
    if (nbsuccs > 0) {
      if (!dstData.isAlreadyIn())
        ctx.push(ForwardPhaseWorkItem(dstID, dstData.distance));
    }
    dstData.unlock();
  }

  void dagConstruction(galois::InsertBag<ForwardPhaseWorkItem>& wl) {
    galois::for_each(galois::iterate(wl),
                     [&](ForwardPhaseWorkItem& wi, auto& ctx) {
                       uint32_t srcID    = wi.nodeID;
                       NodeType& srcData = graph.getData(srcID);
                       srcData.markOut();

                       // loop through all edges
                       for (auto e : graph.edges(srcID)) {
                         BCEdge& edgeData  = graph.getEdgeData(e);
                         uint32_t dstID    = graph.getEdgeDst(e);
                         NodeType& dstData = graph.getData(dstID);

                         if (srcID == dstID)
                           continue; // ignore self loops

                         // lock in set order to prevent deadlock (lower id
                         // first)
                         // TODO run even in serial version; find way to not
                         // need to run
                         if (srcID < dstID) {
                           srcData.lock();
                           dstData.lock();
                         } else {
                           dstData.lock();
                           srcData.lock();
                         }

                         const int elevel = edgeData.level;
                         const int ADist  = srcData.distance;
                         const int BDist  = dstData.distance;

                         if (BDist - ADist > 1) {
                           // Shortest Path + First Update (and Correct Node)
                           this->spAndFU(srcID, dstID, edgeData, ctx);
                         } else if (elevel == ADist && BDist == ADist + 1) {
                           // Update Sigma
                           this->updateSigma(srcID, dstID, edgeData, ctx);
                         } else if (BDist == ADist + 1 && elevel != ADist) {
                           // First Update not combined with Shortest Path
                           this->firstUpdate(srcID, dstID, edgeData, ctx);
                         } else { // No Action
                           noActionCount.update(1);
                           srcData.unlock();
                           dstData.unlock();
                         }
                       }
                     },
                     galois::wl<OBIM>(FPWorkItemIndexer()),
                     galois::no_conflicts(), galois::loopname("ForwardPhase"));
  }

  void dependencyBackProp(galois::InsertBag<uint32_t>& wl) {
    galois::for_each(galois::iterate(wl),
                     [&](uint32_t srcID, auto& ctx) {
                       NodeType& srcData = graph.getData(srcID);
                       srcData.lock();

                       if (srcData.nsuccs == 0) {
                         const double srcDelta = srcData.delta;
                         srcData.bc += srcDelta;

                         srcData.unlock();

                         NodeType::predTY& srcPreds = srcData.preds;

                         // loop through src's predecessors
                         for (unsigned i = 0; i < srcPreds.size(); i++) {
                           uint32_t predID    = srcPreds[i];
                           NodeType& predData = graph.getData(predID);

                           assert(srcData.sigma >= 1);
                           const double term = (double)predData.sigma *
                                               (1.0 + srcDelta) / srcData.sigma;
                           // if (std::isnan(term)) {
                           //  galois::gPrint(predData.sigma, " ", srcDelta, "
                           //  ", srcData.sigma, "\n");
                           //}
                           predData.lock();
                           predData.delta += term;
                           const unsigned prevPdNsuccs = predData.nsuccs;
                           predData.nsuccs--;

                           if (prevPdNsuccs == 1) {
                             predData.unlock();
                             ctx.push(predID);
                           } else {
                             predData.unlock();
                           }
                         }

                         // reset data in preparation for next source
                         srcData.reset();
                         for (auto e : graph.edges(srcID)) {
                           graph.getEdgeData(e).reset();
                         }
                       } else {
                         srcData.unlock();
                       }
                     },
                     galois::no_conflicts(), galois::loopname("BackwardPhase"));
  }

  void findLeaves(galois::InsertBag<uint32_t>& fringeWL, unsigned nnodes) {
    LeafCounter leafCount{"leaf nodes in DAG"};
    galois::do_all(galois::iterate(0u, nnodes),
                   [&](auto i) {
                     NodeType& n = graph.getData(i);

                     if (n.nsuccs == 0 && n.distance < infinity) {
                       leafCount.update(1);
                       fringeWL.push(i);
                     }
                   },
                   galois::loopname("LeafFind"));
  }
};

static const char* name = "Betweenness Centrality";
static const char* desc = "Computes betwenness centrality in an unweighted "
                          "graph";

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, NULL);

  if (BC_CONCURRENT) {
    galois::gInfo("Running in concurrent mode with ", numThreads, " threads");
  } else {
    galois::gInfo("Running in serial mode");
  }

  galois::gInfo("Constructing graph");
  // create bidirectional graph
  Graph bcGraph;

  galois::StatTimer graphConstructTimer("GRAPH_CONSTRUCT");
  graphConstructTimer.start();

  galois::graphs::BufferedGraph<void> fileReader;
  fileReader.loadGraph(filename);
  bcGraph.allocateFrom(fileReader.size(), fileReader.sizeEdges());
  bcGraph.constructNodes();

  galois::do_all(galois::iterate((uint32_t)0, fileReader.size()),
                 [&](uint32_t i) {
                   auto b = fileReader.edgeBegin(i);
                   auto e = fileReader.edgeEnd(i);

                   bcGraph.fixEndEdge(i, *e);

                   while (b < e) {
                     bcGraph.constructEdge(*b, fileReader.edgeDestination(*b));
                     b++;
                   }
                 });
  bcGraph.constructIncomingEdges();

  graphConstructTimer.stop();

  BetweenessCentralityAsync bcExecutor(bcGraph);

  unsigned nnodes = bcGraph.size();
  uint64_t nedges = bcGraph.sizeEdges();
  galois::gInfo("Num nodes is ", nnodes, ", num edges is ", nedges);
  galois::gInfo("Using OBIM chunk size: ", CHUNK_SIZE);
  galois::gInfo("Note that optimal chunk size may differ depending on input "
                "graph");
  galois::runtime::reportStat_Single("BCAsync", "ChunkSize", CHUNK_SIZE);

  galois::reportPageAlloc("MemAllocPre");
  galois::gInfo("Going to pre-allocate pages");
  galois::preAlloc(
      std::min((uint64_t)(std::min(galois::getActiveThreads(), 100u) *
                          std::max((nnodes / 4500000), (unsigned)5) *
                          std::max((nedges / 30000000), (uint64_t)5) * 2.5),
               (uint64_t)1500) +
      5);
  galois::gInfo("Pre-allocation complete");
  galois::reportPageAlloc("MemAllocMid");

  // reset everything in preparation for run
  galois::do_all(galois::iterate(0u, nnodes),
                 [&](auto i) { bcGraph.getData(i).reset(); });
  galois::do_all(galois::iterate(UINT64_C(0), nedges),
                 [&](auto i) { bcGraph.getEdgeData(i).reset(); });

  // reading in list of sources to operate on if provided
  std::ifstream sourceFile;
  std::vector<uint64_t> sourceVector;
  if (sourcesToUse != "") {
    sourceFile.open(sourcesToUse);
    std::vector<uint64_t> t(std::istream_iterator<uint64_t>{sourceFile},
                            std::istream_iterator<uint64_t>{});
    sourceVector = t;
    sourceFile.close();
  }

  if (numOfSources == 0) {
    numOfSources = nnodes;
  }

  // if user does specifes a certain number of out sources (i.e. only sources
  // with outgoing edges), we need to loop over the entire node set to look for
  // good sources to use
  uint32_t goodSource = 0;
  if (numOfOutSources != 0) {
    numOfSources = nnodes;
  }

  // only use at most the number of sources in the passed in source file (if
  // such a file was actually pass in)
  if (sourceVector.size() != 0) {
    if (numOfSources > sourceVector.size()) {
      numOfSources = sourceVector.size();
    }
  }

  galois::InsertBag<ForwardPhaseWorkItem> forwardPhaseWL;
  galois::InsertBag<uint32_t> backwardPhaseWL;

  galois::gInfo("Beginning execution");

  galois::StatTimer executionTimer;
  executionTimer.start();
  for (uint32_t i = 0; i < numOfSources; ++i) {
    uint32_t sourceToUse = i;
    if (sourceVector.size() != 0) {
      sourceToUse = sourceVector[i];
    }

    // ignore nodes with no neighbors
    if (!std::distance(bcGraph.edge_begin(sourceToUse),
                       bcGraph.edge_end(sourceToUse))) {
      galois::gDebug(sourceToUse, " has no outgoing edges");
      continue;
    }

    forwardPhaseWL.push_back(ForwardPhaseWorkItem(sourceToUse, 0));
    NodeType& active = bcGraph.getData(sourceToUse);
    active.initAsSource();
    galois::gDebug("Source is ", sourceToUse);

    bcExecutor.dagConstruction(forwardPhaseWL);
    forwardPhaseWL.clear();

    bcExecutor.findLeaves(backwardPhaseWL, nnodes);

    double backupSrcBC = active.bc;
    bcExecutor.dependencyBackProp(backwardPhaseWL);

    active.bc = backupSrcBC; // current source BC should not get updated

    backwardPhaseWL.clear();

    // break out once number of sources user specified to do (if any) has been
    // reached
    goodSource++;
    if (numOfOutSources != 0 && goodSource >= numOfOutSources)
      break;
  }
  executionTimer.stop();

  galois::gInfo("Number of sources with outgoing edges was ", goodSource);

  galois::reportPageAlloc("MemAllocPost");

  // prints out first 10 node BC values
  if (!skipVerify) {
    int count = 0;
    for (unsigned i = 0; i < nnodes && count < 10; ++i, ++count) {
      galois::gPrint(count, ": ", std::setiosflags(std::ios::fixed),
                     std::setprecision(6), bcGraph.getData(i).bc, "\n");
    }
  }

  if (generateCert) {
    std::cerr << "Writting out bc values...\n";
    std::stringstream outfname;
    outfname << "certificate"
             << "_" << numThreads << ".txt";
    std::string fname = outfname.str();
    std::ofstream outfile(fname.c_str());
    for (unsigned i = 0; i < nnodes; ++i) {
      outfile << i << " " << std::setiosflags(std::ios::fixed)
              << std::setprecision(9) << bcGraph.getData(i).bc << "\n";
    }
    outfile.close();
  }

  return 0;
}
