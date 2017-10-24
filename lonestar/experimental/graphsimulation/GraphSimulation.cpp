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

struct QNode {
  bool matched;
  uint32_t label;
  uint32_t id;
};

template<typename Graph>
void initializeQueryGraph(Graph& g) {
  uint32_t i = 0;
  for (auto n: g) {
    g.getData(n).id = i++;
  }

  galois::do_all(galois::iterate(g),
      [&g] (typename Graph::GraphNode n) {
        auto& data = g.getData(n);
        data.matched = false;
        data.label = data.id;
      },
      galois::no_stats());
}

template<typename Graph>
void initializeDataGraph(Graph& g, uint32_t labelCount) {
  uint32_t i = 0;
  for (auto n: g) {
    g.getData(n).id = i++;
  }

  galois::do_all(galois::iterate(g),
      [&g, labelCount] (typename Graph::GraphNode n) {
        auto& data = g.getData(n);
        data.matched = false;
        data.label = data.id % labelCount; // TODO: change to random
      },
      galois::no_stats());
}

template<typename QG, typename DG, typename W>
void matchLabel(QG& qG, DG& dG, W& w) {
  galois::do_all(galois::iterate(dG),
      [&dG, &qG, &w] (typename DG::GraphNode dn) {
        auto& dData = dG.getData(dn);
        for (auto qn: qG) {
          auto& qData = qG.getData(qn);
          if (qData.label == dData.label) {
            qData.matched = true;
            dData.matched = true;
            dData.matchedQGNode = qn;
            w.push_back(dn);
            break;
          }
        }
      },
      galois::loopname("MatchLabel"),
      galois::timeit());
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
      os << "(" << qG.getData(dData.matchedQGNode).id << ", " << dData.id << ")" << std::endl;
    }
  }

  if (outputFile.size()) {
    ofs.close();
  }
}

void runGraphSimulation() {
  using QGraph = galois::graphs::LC_CSR_Graph<QNode, void>::with_no_lockable<true>::type::with_numa_alloc<true>::type;
  using QGNode = QGraph::GraphNode;

  QGraph qG;
  galois::graphs::readGraph(qG, queryGraph);
  std::cout << "Read query graph of " << qG.size() << " nodes" << std::endl;
  initializeQueryGraph(qG);

  struct DNode {
    bool matched;
    QGNode matchedQGNode;
    uint32_t label;
    uint32_t id;
  };
  using DGraph = galois::graphs::LC_CSR_Graph<DNode, void>::with_no_lockable<true>::type::with_numa_alloc<true>::type;
  using DGNode = DGraph::GraphNode;

  DGraph dG;
  galois::graphs::readGraph(dG, dataGraph);
  std::cout << "Read data graph of " << dG.size() << " nodes" << std::endl;
  initializeDataGraph(dG, qG.size());

  using WorkQueue = galois::InsertBag<DGNode>;
  WorkQueue w[2];
  WorkQueue* cur = &w[0];
  WorkQueue* next = &w[1];

  matchLabel(qG, dG, *next);
  if (existEmptyLabelMatchQGNode(qG)) {
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

          // match children links
          // TODO: sort query edges and data edges by label, e.g. node data
          for (auto qe: qG.edges(dData.matchedQGNode)) {
            auto qDst = qG.getEdgeDst(qe);

            bool matched = false;
            for (auto de: dG.edges(dn)) {
              auto& dDstData = dG.getData(dG.getEdgeDst(de));
              if (dDstData.matched && qDst == dDstData.matchedQGNode) {
                matched = true;
                break;
              }
            }

            // remove dn when we have an unmatched edge
            if (!matched) {
              dData.matched = false;
              break;
            }
          }

          // keep dn for next round
          if (dData.matched) {
            next->push_back(dn);
          }
        },
        galois::loopname("GraphSimulation"),
        galois::timeit());

    sizeCur = std::distance(cur->begin(), cur->end());
    sizeNext = std::distance(next->begin(), next->end());
  }

  reportSimulation(qG, dG);
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

  galois::StatTimer T("OverheadTime");
  T.start();

  switch(simType) {
  case Simulation::graph:
    runGraphSimulation();
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
