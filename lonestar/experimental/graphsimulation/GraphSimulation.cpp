/** Single source shortest paths -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @section Description
 *
 * Graph Simulation.
 *
 * @author Roshan Dathathri <roshan@cs.utexas.edu>
 * @author Yi-Shan Lu <yishanlu@cs.utexas.edu>
 */
#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <fstream>
#include <string>

namespace cll = llvm::cl;

static const char* name = "Graph Simulation";
static const char* desc =
    "Compute graph simulation for a pair of given query and data graphs";
static const char* url = "graph_simulation";

enum Simulation {
  graph,
  dual,
  strong
};

static cll::opt<Simulation> simType("simType",
                                    cll::desc("Type of simulation:"),
                                    cll::values(clEnumValN(Simulation::graph, "graphSim", "keep node labeling + outgoing transitions (default)"),
                                                clEnumValN(Simulation::dual, "dualSim", "graphSim + keep incoming transitions"),
                                                clEnumValN(Simulation::strong, "strongSim", "dualSim + nodes matched within a ball of r = diameter(query graph)"),
                                                clEnumValEnd),
                                    cll::init(Simulation::graph));

static cll::opt<std::string> queryGraph("q",
                                        cll::desc("<query graph>"), 
                                        cll::Required);

static cll::opt<std::string> dataGraph("d",
                                       cll::desc("<data graph>"),
                                       cll::Required);

static cll::opt<std::string> outputFile("o",
                                        cll::desc("[match output]"));

struct GNode {
  uint32_t label;
  uint64_t id; // specific to the label
  uint64_t matched; // maximum of 64 nodes in the query graph
  // TODO: make matched a dynamic bitset
};

typedef galois::graphs::LC_CSR_Graph<GNode, void>::with_no_lockable<true>::type::with_numa_alloc<true>::type Graph;


template<typename G>
void initializeQueryGraph(G& g) {
  uint32_t i = 0;
  for (auto n: g) {
    g.getData(n).id = i++;
  }

  galois::do_all(galois::iterate(g),
      [&g] (typename Graph::GraphNode n) {
        auto& data = g.getData(n);
        data.matched = 0; // matches to none
        data.label = data.id;
      });
}

template<typename G>
void initializeDataGraph(G& g, uint32_t labelCount) {
  uint32_t i = 0;
  for (auto n: g) {
    g.getData(n).id = i++;
  }

  galois::do_all(galois::iterate(g),
      [&g, labelCount] (typename Graph::GraphNode n) {
        auto& data = g.getData(n);
        data.matched = 0; // matches to none
        data.label = data.id % labelCount; // TODO: change to random
      });
}

template<typename QG, typename DG, typename W>
void matchLabel(QG& qG, DG& dG, W& w) {
  galois::do_all(galois::iterate(dG),
      [&dG, &qG, &w] (typename DG::GraphNode dn) {
        auto& dData = dG.getData(dn);
        for (auto qn: qG) {
          assert(qn < 64); // because matched is 64-bit
          auto& qData = qG.getData(qn);
          if (qData.label == dData.label) {
            if (!qData.matched) {
              qData.matched = 1;
            }
            if (!dData.matched) {
              w.push_back(dn);
            }
            dData.matched |= 1 << qn; // multiple matches
          }
        }
      },
      galois::loopname("MatchLabel"));
}

template<typename QG>
bool existEmptyLabelMatchQGNode(QG& qG) {
  for (auto qn: qG) {
    auto& qData = qG.getData(qn);
    if (!qData.matched) {
      std::cout << "empty label match for query node " << qData.id << std::endl;
      return true;
    }
  }
  return false;
}

template<typename QG, typename DG>
void reportSimulation(QG& qG, DG& dG) {
  std::streambuf* buf;
  std::ofstream ofs;

  if (outputFile.size()) {
    ofs.open(outputFile);
    buf = ofs.rdbuf();
  } else {
    buf = std::cout.rdbuf();
  }

  std::ostream os(buf);
  for (auto dn: dG) {
    auto& dData = dG.getData(dn);
    if (dData.matched) {
      for (auto qn: qG) { // multiple matches
        uint64_t mask = (1 << qn);
        if (dData.matched & mask) {
          os << "(" << qG.getData(qn).id << ", " << dData.id << ")" << std::endl;
        }
      }
    }
  }

  if (outputFile.size()) {
    ofs.close();
  }
}

template<typename QG, typename DG>
void runGraphSimulation(QG& qG, DG& dG) {
  using DGNode = Graph::GraphNode;

  using WorkQueue = galois::InsertBag<DGNode>;
  WorkQueue w[2];
  WorkQueue* cur = &w[0];
  WorkQueue* next = &w[1];

  galois::StatTimer T("GraphSimulation");
  T.start();

  matchLabel(qG, dG, *next);
  if (existEmptyLabelMatchQGNode(qG)) {
    T.stop();
    return;
  }

  auto sizeCur = std::distance(cur->begin(), cur->end());
  auto sizeNext = std::distance(next->begin(), next->end());

  // loop until no more data nodes are removed
  while (sizeCur != sizeNext) {
    std::swap(cur, next);
    next->clear();

    galois::do_all(galois::iterate(*cur),
        [&dG, &qG, cur, next] (DGNode dn) {
          auto& dData = dG.getData(dn);

          for (auto qn: qG) { // multiple matches
            uint64_t mask = (1 << qn);
            if (dData.matched & mask) {
              // match children links
              // TODO: sort query edges and data edges by label, e.g. node data
              for (auto qe: qG.edges(qn)) {
                auto qDst = qG.getEdgeDst(qe);

                bool matched = false;
                for (auto de: dG.edges(dn)) {
                  auto& dDstData = dG.getData(dG.getEdgeDst(de));
                  if (dDstData.matched & (1 << qDst)) {
                    matched = true;
                    break;
                  }
                }

                // remove qn from dn when we have an unmatched edge
                if (!matched) {
                  dData.matched &= ~mask;
                  break;
                }
              }
            }
          }

          // keep dn for next round
          if (dData.matched) {
            next->push_back(dn);
          }
        },
        galois::loopname("CheckChildrenLink"));

    sizeCur = std::distance(cur->begin(), cur->end());
    sizeNext = std::distance(next->begin(), next->end());
  }

  T.stop();
  reportSimulation(qG, dG);
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  Graph qG;
  galois::graphs::readGraph(qG, queryGraph);
  std::cout << "Read query graph of " << qG.size() << " nodes" << std::endl;
  initializeQueryGraph(qG);

  Graph dG;
  galois::graphs::readGraph(dG, dataGraph);
  std::cout << "Read data graph of " << dG.size() << " nodes" << std::endl;
  initializeDataGraph(dG, qG.size());

  galois::StatTimer T("TotalTime");
  T.start();

  switch(simType) {
  case Simulation::graph:
    runGraphSimulation(qG, dG);
    break;
  case Simulation::dual:
//    runDualSimulation();
    break;
  case Simulation::strong:
//    runStrongSimulation();
    break;
  default:
    std::cerr << "Unknown algorithm!" << std::endl;
    abort();
  }

  T.stop();

  return 0;
}
