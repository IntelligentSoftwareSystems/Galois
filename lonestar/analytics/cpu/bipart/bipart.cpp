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

#include "bipart.h"
#include "galois/graphs/ReadGraph.h"
#include "galois/Timer.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/graphs/FileGraph.h"
#include "galois/LargeArray.h"

#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <array>
#include <unordered_set>

namespace cll = llvm::cl;

static const char* name = "BIPART";
static const char* desc =
    "Partitions a hypergraph into K parts and minimizing the graph cut";
static const char* url = "BiPart";

static cll::opt<std::string>
    inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<scheduleMode> schedulingMode(
    cll::desc("Choose a inital scheduling mode:"),
    cll::values(clEnumVal(PLD, "PLD"), clEnumVal(PP, "PP"), clEnumVal(WD, "WD"),
                clEnumVal(RI, "RI"), clEnumVal(MRI, "MRI"),
                clEnumVal(MDEG, "MDEG"), clEnumVal(DEG, "DEG"),
                clEnumVal(MWD, "MWD"), clEnumVal(HIS, "HIS"),
                clEnumVal(RAND, "random")),
    cll::init(RAND));

static cll::opt<bool>
    mtxInput("mtxinput",
             cll::desc("Use text mtx files instead of binary galois gr files"),
             cll::init(false));
static cll::opt<bool> weighted("weighted", cll::desc("weighted"),
                               cll::init(false));
static cll::opt<bool>
    verbose("verbose",
            cll::desc("verbose output (debugging mode, takes extra time)"),
            cll::init(false));
static cll::opt<std::string> outfile("outputFile",
                                     cll::desc("output partition file name"));
static cll::opt<std::string>
    orderedfile("ordered", cll::desc("output ordered graph file name"));
static cll::opt<std::string>
    permutationfile("permutation", cll::desc("output permutation file name"));
static cll::opt<unsigned> csize(cll::Positional,
                                cll::desc("<size of coarsest graph>"),
                                cll::init(25));

static cll::opt<unsigned> refiter(cll::Positional,
                                  cll::desc("<number of iterations in ref>"),
                                  cll::init(2));
static cll::opt<unsigned> numPartitions(cll::Positional,
                                        cll::desc("<number of partitions>"),
                                        cll::init(2));
static cll::opt<double> imbalance(
    "balance",
    cll::desc("Percentage deviated from mean partition size (default 5)"),
    cll::init(5.0));

//! Flag that forces user to be aware that they should be passing in a
//! hMetis graph.
static cll::opt<bool>
    hMetisGraph("hMetisGraph",
                cll::desc("Specify that the input graph is a hMetis"),
                cll::init(false));

static cll::opt<bool>
    output("output", cll::desc("Specify if partitions need to be written"),
           cll::init(false));
double Ctime = 0.0f;
double Ptime = 0.0f;
double Rtime = 0.0f;
/**
 * Partitioning
 */
void Partition(MetisGraph* metisGraph, unsigned coarsenTo, unsigned K) {
  galois::StatTimer execTime("Timer_0");
  execTime.start();

  galois::StatTimer T("CoarsenSEP");
  T.start();
  MetisGraph* mcg = coarsen(metisGraph, coarsenTo, schedulingMode);
  T.stop();

  galois::StatTimer T2("PartitionSEP");
  T2.start();
  partition(mcg, K);
  T2.stop();

  galois::StatTimer T3("Refine");
  T3.start();
  refine(mcg, K, imbalance);
  T3.stop();
  Ctime += (T.get()/1000.0f);
  Ptime += (T2.get()/1000.0f);
  Rtime += (T3.get()/1000.0f);

  execTime.stop();
}

int computingCut(GGraph& g) {

  GNodeBag bag;
  galois::GAccumulator<unsigned> edgecut;
  galois::do_all(
      galois::iterate((size_t)0, g.hedges),
      [&](GNode n) {
        std::set<unsigned> nump;
        for (auto cell : g.edges(n)) {
          auto c   = g.getEdgeDst(cell);
          int part = g.getData(c).getPart();
          nump.insert(part);
        }
        edgecut += (nump.size() - 1);
      },
      galois::loopname("cutsize"));
  return edgecut.reduce();
}

int computingBalance(GGraph& g) {
  int max = 0;
  std::vector<int> parts(numPartitions, 0);
  for (size_t c = g.hedges; c < g.size(); c++) {
    unsigned pp = g.getData(c).getPart();
    parts[pp]++;
  }
  for (unsigned i = 0; i <numPartitions; i++) {
    if (parts[i] > max)
      max = parts[i];
  }
  return max;
}
// printGraphBeg(*graph)

typedef galois::graphs::FileGraph FG;
typedef FG::GraphNode FN;
template <typename GNode, typename Weights>
struct order_by_degree {
  GGraph& graph;
  Weights& weights;
  order_by_degree(GGraph& g, Weights& w) : graph(g), weights(w) {}
  bool operator()(const GNode& a, const GNode& b) {
    uint64_t wa = weights[a];
    uint64_t wb = weights[b];
    int pa      = graph.getData(a, galois::MethodFlag::UNPROTECTED).getPart();
    int pb      = graph.getData(b, galois::MethodFlag::UNPROTECTED).getPart();
    if (pa != pb) {
      return pa < pb;
    }
    return wa < wb;
  }
};

typedef galois::substrate::PerThreadStorage<std::map<GNode, uint64_t>>
    PerThreadDegInfo;

std::map<uint64_t, uint64_t>
cellToNet(std::map<uint64_t, std::vector<uint64_t>> netToCell) {
  std::map<uint64_t, uint64_t> celltonet;
  for (auto n : netToCell) {
    for (auto c : n.second) {
      celltonet[c]++;
    }
  }
  return celltonet;
}

int hash(unsigned val) {
  unsigned long int seed = val * 1103515245 + 12345;
  return ((unsigned)(seed / 65536) % 32768);
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;

  LonestarStart(argc, argv, name, desc, url, &inputFile);

  galois::StatTimer totalTime("TimerTotal");
  totalTime.start();

  if (!hMetisGraph) {
    GALOIS_DIE("This application requires a hMetis graph input;"
               " please use the -hMetisGraph flag "
               " to indicate the input is a hMetisGraph graph.");
  }

  MetisGraph metisGraph;
  GGraph& graph = *metisGraph.getGraph();
  std::ifstream f(inputFile.c_str());
  std::string line;
  std::getline(f, line);
  std::stringstream ss(line);
  uint32_t i1;
  uint64_t i2;
  ss >> i1 >> i2;
  uint32_t hedges = i1;
  uint64_t nodes  = i2;
  std::cout << "hedges: " << hedges << "\n";
  std::cout << "nodes: " << nodes << "\n\n";

  galois::StatTimer T("buildingG");
  T.start();

  galois::gstl::Vector<galois::PODResizeableArray<uint32_t>> edges_id(hedges +
                                                                      nodes);
  std::vector<std::vector<EdgeTy>> edges_data(hedges + nodes);
  std::vector<uint64_t> prefix_edges(nodes + hedges);
  uint32_t cnt   = 0;
  uint32_t edges = 0;
  while (std::getline(f, line)) {
    if (cnt >= hedges) {
      printf("ERROR: too many lines in input file\n");
      exit(-1);
    }
    std::stringstream ss(line);
    int val;
    while (ss >> val) {
      if ((val < 1) || (val > static_cast<long>(nodes))) {
        printf("ERROR: node value %d out of bounds\n", val);
        exit(-1);
      }
      unsigned newval = hedges + (val - 1);
      edges_id[cnt].push_back(newval);
      edges++;
    }
    cnt++;
  }
  f.close();
  graph.hedges = hedges;
  graph.hnodes = nodes;
  std::cout << "number of hedges " << hedges << "\n";
  uint32_t sizes = hedges + nodes;

  galois::do_all(galois::iterate(uint32_t{0}, sizes),
                 [&](uint32_t c) { prefix_edges[c] = edges_id[c].size(); });

  for (uint64_t c = 1; c < nodes + hedges; ++c) {
    prefix_edges[c] += prefix_edges[c - 1];
  }

  graph.constructFrom(nodes + hedges, edges, prefix_edges, edges_id,
                      edges_data);
  galois::do_all(
      galois::iterate(graph),
      [&](GNode n) {
        if (n < hedges)
          graph.getData(n).netnum = n + 1;
        else
          graph.getData(n).netnum = INT_MAX;
        graph.getData(n).netrand = INT_MAX;
        graph.getData(n).netval  = INT_MAX;
        graph.getData(n).nodeid  = n + 1;
      },
      galois::steal(), galois::loopname("build initial graph"));
  T.stop();
  std::cout << "time to build a graph " << T.get() << "\n";
  graphStat(graph);
  std::cout << "\n";
  galois::preAlloc(galois::runtime::numPagePoolAllocTotal() * 10);
  galois::reportPageAlloc("MeminfoPre");
  galois::do_all(
      galois::iterate(graph.hedges, graph.size()),
      [&](GNode item) {
        // accum += g->getData(item).getWeight();
        graph.getData(item, galois::MethodFlag::UNPROTECTED)
            .initRefine(0, true);
        graph.getData(item, galois::MethodFlag::UNPROTECTED).initPartition();
      },
      galois::steal(), galois::loopname("initPart"));

  Partition(&metisGraph, csize, numPartitions);

  const int k = numPartitions;
  // calculating number of iterations/levels required
  int num = log2(k);

  int kValue[k];
  for (int i = 0; i < k; i++)
    kValue[i] = 0;

  kValue[0]           = (k + 1) / 2;
  kValue[(k + 1) / 2] = k / 2;

  galois::do_all(
      galois::iterate((uint64_t)graph.hedges, graph.size()),
      [&](GNode n) {
        unsigned pp = graph.getData(n).getPart();
        if (pp == 1) {
          graph.getData(n).setPart((k + 1) / 2);
        }
      },
      galois::steal(), galois::loopname("set part (original graph)"));

  // running it level by level

  // toProcess contains nodes to be executed in a given level
  std::set<int> toProcess;
  std::set<int> toProcessNew;
  toProcess.insert(0);
  toProcess.insert((k + 1) / 2);

  std::vector<std::vector<GNode>> nodesvec(k);
  // std::array<std::vector<GNode>, 100> hedgesvec;

  for (int level = 0; level < num; level++) {

    for (int i = 0; i < k; i++)
      nodesvec[i].clear();

    // distributing nodes in relevant vectors according to their current
    // partition assignment
    for (GNode n = graph.hedges; n < graph.size(); n++) {
      unsigned pp = graph.getData(n).getPart();
      nodesvec[pp].push_back(n);
    }

    std::vector<std::vector<GNode>> hedgevec(k);

    // distribute hyperedges according to their current partition
    galois::do_all(
        galois::iterate((uint64_t)0, graph.hedges),
        [&](GNode h) {
          auto edge = *(graph.edges(h).begin());
          auto dst  = graph.getEdgeDst(edge);
          auto ii   = graph.getData(dst).getPart();

          bool flag = true;

          for (auto n : graph.edges(h)) {
            auto part = graph.getData(graph.getEdgeDst(n)).getPart();

            if (part != ii) {
              flag = false;
              break;
            }
          }

          if (flag)
            graph.getData(h).setPart(ii);
          else
            graph.getData(h).setPart(100000);
        },
        galois::steal(), galois::loopname("distribute hedges"));

    for (GNode h = 0; h < graph.hedges; h++) {
      unsigned part = graph.getData(h).getPart();
      if (part != 100000)
        hedgevec[part].push_back(h);
    }

    // calling Partition for each partition number
    for (unsigned i : toProcess) {
      if (kValue[i] > 1) {
        MetisGraph metisG;
        GGraph& gr = *metisG.getGraph();

        unsigned ed = 0;

        for (auto h : hedgevec[i])
          graph.getData(h).index = ed++;

        unsigned id = ed;
        for (auto n : nodesvec[i]) {
          graph.getData(n).index = id++;
        }

        unsigned totalnodes = id;
        galois::gstl::Vector<galois::PODResizeableArray<uint32_t>> edges_ids(
            totalnodes);
        std::vector<std::vector<EdgeTy>> edge_data(totalnodes);
        std::vector<uint64_t> pre_edges(totalnodes);
        unsigned edges = 0;

        galois::do_all(
            galois::iterate(hedgevec[i]),
            [&](GNode h) {
              for (auto v : graph.edges(h)) {
                auto vv = graph.getEdgeDst(v);

                uint32_t newid = graph.getData(h).index;
                unsigned nm    = graph.getData(vv).index;
                edges_ids[newid].push_back(nm);
              }
            },
            galois::steal(), galois::loopname("populate edge ids"));

        uint64_t num_edges_acc = 0;
        //galois::do_all(
          //  galois::iterate(uint32_t{0}, totalnodes),
            for(uint32_t c = 0;c<totalnodes;c++) {
              pre_edges[c] = edges_ids[c].size();
              num_edges_acc += pre_edges[c];
            }
            //galois::steal(), galois::loopname("set pre edges"));

        edges = num_edges_acc;

        for (uint64_t c = 1; c < totalnodes; ++c) {
          pre_edges[c] += pre_edges[c - 1];
        }
        gr.constructFrom(totalnodes, edges, pre_edges, edges_ids, edge_data);

        gr.hedges = ed;
        gr.hnodes = id - ed;

        galois::do_all(
            galois::iterate(gr),
            [&](GNode n) {
              if (n < gr.hedges)
                gr.getData(n).netnum = n + 1;
              else
                gr.getData(n).netnum = INT_MAX;
              gr.getData(n).netrand = INT_MAX;
              gr.getData(n).netval  = INT_MAX;
              gr.getData(n).nodeid  = n + 1;
            },
            galois::steal(), galois::loopname("build graph: recursion level"));

        Partition(&metisG, csize, kValue[i]);

        MetisGraph* mcg = &metisG;

        // now free up the memory by deleting all coarsened graphs
        while (mcg->getCoarserGraph() != NULL) {
          mcg = mcg->getCoarserGraph();
        }

        while (mcg->getFinerGraph() != NULL &&
               mcg->getFinerGraph()->getFinerGraph() != NULL) {
          mcg = mcg->getFinerGraph();
          delete mcg->getCoarserGraph();
        }

        int tmp                   = kValue[i];
        kValue[i]                 = (tmp + 1) / 2;
        kValue[i + (tmp + 1) / 2] = (tmp) / 2;
        toProcessNew.insert(i);
        toProcessNew.insert(i + (tmp + 1) / 2);

        galois::do_all(
            galois::iterate(nodesvec[i]),
            [&](GNode v) {
              GNode n     = graph.getData(v).index;
              unsigned pp = gr.getData(n).getPart();
              if (pp == 0) {
                graph.getData(v).setPart(i);
              } else if (pp == 1) {
                graph.getData(v).setPart(i + (tmp + 1) / 2);
              }
            },
            galois::steal(),
            galois::loopname("set part: inside recursive call"));

        delete mcg;
      } // end if
    }   // end for

    toProcess = toProcessNew;
    toProcessNew.clear();
  } // end while
  std::cout<<"Coarsening time(s):,"<<Ctime<<"\n";
  std::cout<<"Partitiong time(s):,"<<Ptime<<"\n";
  std::cout<<"Refinement time(s):,"<<Rtime<<"\n";
  std::cout<<"\n";
  std::cout<<"Edge Cut,"<<computingCut(graph)<<"\n\n";

  galois::runtime::reportStat_Single("BiPart", "Edge Cut", computingCut(graph));
  //galois::runtime::reportStat_Single("BiPart", "zero-one",
  //                                   computingBalance(graph));

  totalTime.stop();
  if (output) {

    std::vector<std::vector<uint64_t> >parts(numPartitions);

    for (GNode n = graph.hedges; n < graph.size(); n++) {
      unsigned p = graph.getData(n).getPart();
      parts[p].push_back(n - graph.hedges + 1);
    }

    std::ofstream outputFile(outfile.c_str());

    for (unsigned i = 0; i < numPartitions; i++) {
      outputFile << i+1 << " ";
      for (auto v : parts[i]) 
        outputFile << v << " "; 
      outputFile << "\n";
    }
    outputFile.close();
  }
  return 0;
}
