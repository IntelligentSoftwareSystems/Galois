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
#include "GraphSimulation.h"

#include <iostream>
#include <fstream>

template<typename QG, typename DG, typename W>
void matchLabel(QG& qG, DG& dG, W& w) {
  galois::do_all(galois::iterate(dG.begin(), dG.end()),
      [&qG, &dG, &w] (typename DG::GraphNode dn) {
        auto& dData = dG.getData(dn);
        dData.matched = 0; // matches to none
        for (auto qn: qG) {
          assert(qn < 64); // because matched is 64-bit
          auto& qData = qG.getData(qn);
          if (qData.label & dData.label) { // query could be any or multiple labels
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

void reportGraphSimulation(Graph& qG, Graph& dG, std::string outputFile) {
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

void runGraphSimulation(Graph& qG, Graph& dG) {
  using DGNode = Graph::GraphNode;

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

          for (auto qn: qG) { // multiple matches
            uint64_t mask = (1 << qn);
            if (dData.matched & mask) {
              // match children links
              // TODO: sort query edges and data edges by label; e.g., node data
#ifdef QUERY_GRAPH_GENERAL_SOLUTION
              for (auto qe: qG.edges(qn)) {
                auto qeData = qG.getEdgeData(qe);
                auto qDst = qG.getEdgeDst(qe);

                bool matched = false;
                for (auto de: dG.edges(dn)) {
                  auto deData = dG.getEdgeData(de);
                  if (qeData.label & deData.label) { // query could be any or multiple labels
                    auto& dDstData = dG.getData(dG.getEdgeDst(de));
                    if (dDstData.matched & (1 << qDst)) {
                      matched = true;
                      break;
                    }
                  }
                }

                // remove qn from dn when we have an unmatched edge
                if (!matched) {
                  dData.matched &= ~mask;
                  break;
                }
              }
              // TODO: compare matched edges for timestamp and dst-id inequality
#else
              // assume query graph has at the most 2 edges for any node
              auto qe1 = qG.edge_begin(qn);
              auto qend = qG.edge_end(qn);
              if (qe1 != qend) {
                auto& qeData = qG.getEdgeData(qe1);
                auto qDst = qG.getEdgeDst(qe1);

                bool matched = false;
                for (auto& de: dG.edges(dn)) {
                  auto& deData = dG.getEdgeData(de);
                  if (qeData.label & deData.label) { // query could be any or multiple labels
                    auto dDst = dG.getEdgeDst(de);
                    auto& dDstData = dG.getData(dDst);
                    if (dDstData.matched & (1 << qDst)) {

                      auto qe2 = qe1 + 1;
                      if (qe2 == qend) { // only 1 edge
                        matched = true;
                        break;
                      } else {
                        assert ((qe2 + 1) == qend);
                        // match the second edge
                        auto& qeData2 = qG.getEdgeData(qe2);
                        auto qDst2 = qG.getEdgeDst(qe2);

                        for (auto& de2: dG.edges(dn)) {
                          auto& deData2 = dG.getEdgeData(de2);
                          if (qeData2.label & deData2.label) { // query could be any or multiple labels
                            auto dDst2 = dG.getEdgeDst(de2);
                            auto& dDstData2 = dG.getData(dDst2);
                            if (dDstData2.matched & (1 << qDst2)) {
                              assert(qeData.timestamp != qeData2.timestamp);
                              if ((qeData.timestamp <= qeData2.timestamp) == (deData.timestamp <= deData2.timestamp)) {
                                if ((qDst != qDst2) == (dDst != dDst2)) {
                                  matched = true;
                                  break;
                                }
                              }
                            }
                          }
                        }

                        if (matched) break;
                      }
                    }
                  }
                }

                // remove qn from dn when we have an unmatched edge
                if (!matched) {
                  dData.matched &= ~mask;
                  break;
                }
              }
#endif
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
}
