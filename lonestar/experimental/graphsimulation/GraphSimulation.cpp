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
#ifdef UNIQUE_QUERY_NODES
                                if ((qDst != qDst2) == (dDst != dDst2)) {
#endif
                                  matched = true;
                                  break;
#ifdef UNIQUE_QUERY_NODES
                                }
#endif
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

unsigned rightmostSetBitPos(uint32_t n) {
  assert(n != 0);
  if (n & 1) return 0;

  // unset rightmost bit and xor with itself
  n = n ^ (n & (n - 1));

  unsigned pos = 0;
  while (n) {
    n >>= 1;
    pos++;
  }
  return pos-1;
}

void reportGraphSimulation(AttributedGraph& qG, AttributedGraph& dG, std::string outputFile) {
  std::streambuf* buf;
  std::ofstream ofs;

  if (outputFile.size()) {
    ofs.open(outputFile);
    buf = ofs.rdbuf();
  } else {
    buf = std::cout.rdbuf();
  }

  std::ostream os(buf);

  Graph& qgraph = qG.graph;
  auto& qnodeNames = qG.nodeNames;
  Graph& graph = dG.graph;
  auto& nodeLabelNames = dG.nodeLabelNames;
  auto& edgeLabelNames = dG.edgeLabelNames;
  auto& nodeNames = dG.nodeNames;
  for(auto n: graph) {
    auto& src = graph.getData(n);
    auto& srcLabel = nodeLabelNames[rightmostSetBitPos(src.label)];
    auto& srcName = nodeNames[src.id];
    for(auto e: graph.edges(n)) {
      auto& dst = graph.getData(graph.getEdgeDst(e));
      auto& dstLabel = nodeLabelNames[rightmostSetBitPos(dst.label)];
      auto& dstName = nodeNames[dst.id];
      auto& ed = graph.getEdgeData(e);
      auto& edgeLabel = edgeLabelNames[rightmostSetBitPos(ed.label)];
      auto& edgeTimestamp = ed.timestamp;
      for(auto qn: qgraph) {
        uint64_t mask = (1 << qn);
        if (src.matched & mask) {
          for(auto qe: qgraph.edges(qn)) {
            auto& qeData = qgraph.getEdgeData(qe);
            if (qeData.label & ed.label) { // query could be any or multiple labels
              auto qDst = qgraph.getEdgeDst(qe);
              mask = (1 << qDst);
              if (dst.matched & mask) {
                auto& qSrcName = qnodeNames[qgraph.getData(qn).id];
                auto& qDstName = qnodeNames[qgraph.getData(qDst).id];
                os << srcLabel << " " << srcName << " ("  << qSrcName << ") "
                   << edgeLabel << " " << dstLabel << " "
                   << dstName << " ("  << qDstName << ") "
                   << " at " << edgeTimestamp << std::endl;
                break;
              }
            }
          }
        }
      }
    }
  }

  if (outputFile.size()) {
    ofs.close();
  }
}

void matchNodeWithRepeatedActions(Graph &graph, uint32_t nodeLabel, uint32_t action) {
  // initialize matched
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (typename Graph::GraphNode n) {
      auto& data = graph.getData(n);
      data.matched = 0; // matches to none
    },
    galois::loopname("InitMatched"));

  // match nodes
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (typename Graph::GraphNode n) {
      auto& data = graph.getData(n);
      if (data.label == nodeLabel) {
        unsigned numActions = 0;
        Graph::GraphNode prev = 0;
        for (auto e: graph.edges(n)) {
          auto& eData = graph.getEdgeData(e);
          if (eData.label == action) {
            ++numActions;
            if (numActions == 1) {
              prev = graph.getEdgeDst(e);
            } else {
              if (prev != graph.getEdgeDst(e)) {
                data.matched = 1;
                break;
              }
            }
          }
        }
      }
    },
    galois::loopname("MatchNodes"));

  // match destination of matched nodes
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (typename Graph::GraphNode n) {
      auto& data = graph.getData(n);
      if (data.matched & 1) {
        for (auto e: graph.edges(n)) {
          auto& eData = graph.getEdgeData(e);
          if (eData.label == action) {
            auto dst = graph.getEdgeDst(e);
            auto& dstData = graph.getData(dst);
            dstData.matched |= 2; // atomicity not required
          }
        }
      }
    },
    galois::loopname("MatchNodesDsts"));
}

void matchNodeWithTwoActions(Graph &graph, uint32_t nodeLabel, uint32_t action1, uint32_t dstNodeLabel1, uint32_t action2, uint32_t dstNodeLabel2) {
  // initialize matched
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (typename Graph::GraphNode n) {
      auto& data = graph.getData(n);
      data.matched = 0; // matches to none
    },
    galois::loopname("InitMatched"));

  // match nodes
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (typename Graph::GraphNode n) {
      auto& data = graph.getData(n);
      if (data.label == nodeLabel) {
        bool foundAction1 = false;
        bool foundAction2 = false;
        for (auto e: graph.edges(n)) {
          auto& eData = graph.getEdgeData(e);
          bool mayAction1 = (eData.label == action1);
          bool mayAction2 = (eData.label == action2);
          if (mayAction1 || mayAction2) {
            auto dst = graph.getEdgeDst(e);
            auto& dstData = graph.getData(dst);
            if (mayAction1 && (dstData.label == dstNodeLabel1)) {
              foundAction1 = true;
            } else if (mayAction2 && (dstData.label == dstNodeLabel2)) {
              foundAction2 = true;
            }
          }
        }
        if (foundAction1 && foundAction2) {
          data.matched = 1;
        }
      }
    },
    galois::loopname("MatchNodes"));

  // match destination of matched nodes
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (typename Graph::GraphNode n) {
      auto& data = graph.getData(n);
      if (data.matched & 1) {
        for (auto e: graph.edges(n)) {
          auto& eData = graph.getEdgeData(e);
          bool mayAction1 = (eData.label == action1);
          bool mayAction2 = (eData.label == action2);
          if (mayAction1 || mayAction2) {
            auto dst = graph.getEdgeDst(e);
            auto& dstData = graph.getData(dst);
            if (mayAction1 && (dstData.label == dstNodeLabel1)) {
              dstData.matched |= 2; // atomicity not required
            } else if (mayAction2 && (dstData.label == dstNodeLabel2)) {
              dstData.matched |= 4; // atomicity not required
            }
          }
        }
      }
    },
    galois::loopname("MatchNodesDsts"));
}

size_t countMatchedNodes(Graph& graph) {
  galois::GAccumulator<size_t> numMatched;
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (typename Graph::GraphNode n) {
      auto& data = graph.getData(n);
      if (data.matched) {
        numMatched += 1;
      }
    },
    galois::loopname("CountMatchedNodes"));
  return numMatched.reduce();
}

void returnMatchedNodes(AttributedGraph& dataGraph, MatchedNode* matchedNodes) {
  Graph& graph = dataGraph.graph;
  auto& nodeLabelNames = dataGraph.nodeLabelNames;
  auto& nodeNames = dataGraph.nodeNames;

  size_t i = 0;
  for (auto n: graph) {
    auto& data = graph.getData(n);
    if (data.matched) {
      matchedNodes[i].id = data.id;
      matchedNodes[i].label = nodeLabelNames[data.label].c_str();
      matchedNodes[i].name = nodeNames[n].c_str();
      ++i;
    }
  }
}

void reportMatchedNodes(AttributedGraph &dataGraph, std::string outputFile) {
  Graph& graph = dataGraph.graph;
  auto& nodeLabelNames = dataGraph.nodeLabelNames;
  auto& nodeNames = dataGraph.nodeNames;

  std::streambuf* buf;
  std::ofstream ofs;

  if (outputFile.size()) {
    ofs.open(outputFile);
    buf = ofs.rdbuf();
  } else {
    buf = std::cout.rdbuf();
  }

  std::ostream os(buf);

  os << "Matched nodes:\n";
  for (auto n: graph) {
    auto& data = graph.getData(n);
    if (data.matched) {
      os << nodeLabelNames[data.label] << " " << nodeNames[n] << std::endl;
    }
  }

  if (outputFile.size()) {
    ofs.close();
  }
}

void matchNeighbors(Graph& graph, uint32_t uuid, uint32_t nodeLabel, uint32_t action, uint32_t neighborLabel) {
  // initialize matched
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (typename Graph::GraphNode n) {
      auto& data = graph.getData(n);
      data.matched = 0; // matches to none
    },
    galois::loopname("InitMatched"));

  // match destinations of node
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (typename Graph::GraphNode n) {
      auto& data = graph.getData(n);
      if (data.id == uuid) {
        assert(data.label == nodeLabel);
        for (auto e: graph.edges(n)) {
          auto& eData = graph.getEdgeData(e);
          if (eData.label == action) {
            auto dst = graph.getEdgeDst(e);
            auto& dstData = graph.getData(dst);
            if (dstData.label == neighborLabel) {
              dstData.matched |= 1; // atomicity not required
            }
          }
        }
      }
    },
    galois::loopname("MatchNodesDsts"));
}

size_t countMatchedNeighbors(Graph& graph, uint32_t uuid) {
  galois::GAccumulator<size_t> numMatched;
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (typename Graph::GraphNode n) {
      auto& data = graph.getData(n);
      if (data.matched) {
        numMatched += 1;
      }
    },
    galois::loopname("CountMatchedNeighbors"));
  return numMatched.reduce();
}

void returnMatchedNeighbors(AttributedGraph& dataGraph, uint32_t uuid, MatchedNode* matchedNeighbors) {
  Graph& graph = dataGraph.graph;
  auto& nodeLabelNames = dataGraph.nodeLabelNames;
  auto& nodeNames = dataGraph.nodeNames;

  size_t i = 0;
  for (auto n: graph) {
    auto& data = graph.getData(n);
    if (data.matched) {
      matchedNeighbors[i].id = data.id;
      matchedNeighbors[i].label = nodeLabelNames[data.label].c_str();
      matchedNeighbors[i].name = nodeNames[n].c_str();
      ++i;
    }
  }
}

void reportMatchedNeighbors(AttributedGraph &dataGraph, uint32_t uuid, std::string outputFile) {
  Graph& graph = dataGraph.graph;
  auto& nodeLabelNames = dataGraph.nodeLabelNames;
  auto& nodeNames = dataGraph.nodeNames;

  std::streambuf* buf;
  std::ofstream ofs;

  if (outputFile.size()) {
    ofs.open(outputFile);
    buf = ofs.rdbuf();
  } else {
    buf = std::cout.rdbuf();
  }

  std::ostream os(buf);

  os << "Matched nodes:\n";
  for (auto n: graph) {
    auto& data = graph.getData(n);
    if (data.matched) {
      os << nodeLabelNames[data.label] << " " << nodeNames[n] << std::endl;
    }
  }

  if (outputFile.size()) {
    ofs.close();
  }
}
