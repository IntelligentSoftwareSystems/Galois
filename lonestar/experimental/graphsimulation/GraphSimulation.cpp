#include "GraphSimulation.h"

template<typename QG, typename DG, typename W>
void matchLabel(QG& qG, DG& dG, W& w) {
  galois::do_all(galois::iterate(dG.begin(), dG.end()),
      [&qG, &dG, &w] (auto dn) {
        auto& dData = dG.getData(dn);
        dData.matched = 0; // matches to none
        for (auto qn: qG) {
          assert(qn < 64); // because matched is 64-bit
          auto& qData = qG.getData(qn);
          if (qData.label == dData.label) { // TODO: query could be any or multiple labels
            if (!qData.matched) {
              qData.matched = 1;
            }
            if (!dData.matched) {
              w.push_back(dn);
            }
            dData.matched |= 1 << qn; // multiple matches
          }
        }
        for (auto de: dG.edges(dn)) {
          auto& deData = dG.getEdgeData(de);
          deData.matched = 0; // matches to none
        }
      },
      galois::loopname("MatchLabel"));
}

template<typename QG>
bool existEmptyLabelMatchQGNode(QG& qG) {
  for (auto qn: qG) {
    auto& qData = qG.getData(qn);
    if (!qData.matched) {
      //std::cout << "No label matched for query node " << qData.id << std::endl;
      return true;
    }
  }
  return false;
}

template<bool useLimit, bool useWindow, bool queryNodeHasMoreThan2Edges>
void matchNodesOnce(Graph& qG, Graph& dG, galois::InsertBag<Graph::GraphNode>* cur, galois::InsertBag<Graph::GraphNode>* next, EventLimit limit, EventWindow window) {
  galois::do_all(galois::iterate(*cur),
    [&] (auto dn) {
      auto& dData = dG.getData(dn);

      for (auto qn: qG) { // multiple matches
        uint64_t mask = (1 << qn);
        if (dData.matched & mask) {
          // match children links
          // TODO: sort data edges by timestamp
          // Assumption: query edges are sorted by timestamp
          if (queryNodeHasMoreThan2Edges) {
            uint64_t qPrevEdgeTimestamp = 0;
            uint64_t dPrevEdgeTimestamp = 0;
            for (auto qe: qG.edges(qn)) {
              auto qeData = qG.getEdgeData(qe);
              auto qDst = qG.getEdgeDst(qe);

              bool matched = false;
              uint64_t dNextEdgeTimestamp = std::numeric_limits<uint64_t>::max();
              for (auto de: dG.edges(dn)) {
                auto& deData = dG.getEdgeData(de);
                if (useWindow) {
                  if ((deData.timestamp > window.endTime) || (deData.timestamp < window.startTime)) {
                    continue; // skip this edge since it is not in the time-span of interest
                  }
                }
                if (qeData.label == deData.label) { // TODO: query could be any or multiple labels
                  auto& dDstData = dG.getData(dG.getEdgeDst(de));
                  if (dDstData.matched & (1 << qDst)) {
                    if ((qPrevEdgeTimestamp <= qeData.timestamp) == (dPrevEdgeTimestamp <= deData.timestamp)) {
                      if (dNextEdgeTimestamp > deData.timestamp) {
                        dNextEdgeTimestamp = deData.timestamp; // minimum of matched edges
                      }
                      matched = true;
                    }
                  }
                }
              }

              // remove qn from dn when we have an unmatched edge
              if (!matched) {
                dData.matched &= ~mask;
                break;
              }

              qPrevEdgeTimestamp = qeData.timestamp;
              dPrevEdgeTimestamp = dNextEdgeTimestamp;
            }
          } else {
            // assume query graph has at the most 2 edges for any node
            auto qe1 = qG.edge_begin(qn);
            auto qend = qG.edge_end(qn);
            if (qe1 != qend) {
              auto& qeData = qG.getEdgeData(qe1);
              auto qDst = qG.getEdgeDst(qe1);

              bool matched = false;
              for (auto& de: dG.edges(dn)) {
                auto& deData = dG.getEdgeData(de);
                if (useWindow) {
                  if ((deData.timestamp > window.endTime) || (deData.timestamp < window.startTime)) {
                    continue; // skip this edge since it is not in the time-span of interest
                  }
                }
                if (qeData.label == deData.label) { // TODO: query could be any or multiple labels
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
                        if (qeData2.label == deData2.label) { // TODO: query could be any or multiple labels
                          auto dDst2 = dG.getEdgeDst(de2);
                          auto& dDstData2 = dG.getData(dDst2);
                          if (dDstData2.matched & (1 << qDst2)) {
                            assert(qeData.timestamp != qeData2.timestamp);
                            if (useWindow) {
                              if ((deData2.timestamp > window.endTime) || (deData2.timestamp < window.startTime)) {
                                continue; // skip this edge since it is not in the time-span of interest
                              }
                            }
                            if ((qeData.timestamp <= qeData2.timestamp) == (deData.timestamp <= deData2.timestamp)) {
                              if (useLimit) {
                                if (std::abs(deData.timestamp - deData2.timestamp) > limit.time) {
                                  continue; // skip this sequence of events because too much time has lapsed between them
                                }
                              }
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
          }
          // TODO: add support for dst-id inequality
        }
      }

      // keep dn for next round
      if (dData.matched) {
        next->push_back(dn);
      }
    },
    galois::loopname("MatchNeighbors"));
}

void runGraphSimulation(Graph& qG, Graph& dG, EventLimit limit, EventWindow window, bool queryNodeHasMoreThan2Edges) {
  using WorkQueue = galois::InsertBag<Graph::GraphNode>;
  WorkQueue w[2];
  WorkQueue* cur = &w[0];
  WorkQueue* next = &w[1];

  matchLabel(qG, dG, *next);
  if (existEmptyLabelMatchQGNode(qG)) {
    galois::do_all(galois::iterate(dG.begin(), dG.end()),
        [&qG, &dG, &w] (auto dn) {
          auto& dData = dG.getData(dn);
          dData.matched = 0; // matches to none
      },
      galois::loopname("ResetMatched"));
    return;
  }

  auto sizeCur = std::distance(cur->begin(), cur->end());
  auto sizeNext = std::distance(next->begin(), next->end());

  // loop until no more data nodes are removed
  while (sizeCur != sizeNext) {
    std::swap(cur, next);
    next->clear();

    if (limit.valid) {
      if (window.valid) {
        if (queryNodeHasMoreThan2Edges) {
          matchNodesOnce<true, true, true>(qG, dG, cur, next, limit, window);
        } else {
          matchNodesOnce<true, true, false>(qG, dG, cur, next, limit, window);
        }
      } else {
        if (queryNodeHasMoreThan2Edges) {
          matchNodesOnce<true, false, true>(qG, dG, cur, next, limit, window);
        } else {
          matchNodesOnce<true, false, false>(qG, dG, cur, next, limit, window);
        }
      }
    } else {
      if (window.valid) {
        if (queryNodeHasMoreThan2Edges) {
          matchNodesOnce<false, true, true>(qG, dG, cur, next, limit, window);
        } else {
          matchNodesOnce<false, true, false>(qG, dG, cur, next, limit, window);
        }
      } else {
        if (queryNodeHasMoreThan2Edges) {
          matchNodesOnce<false, false, true>(qG, dG, cur, next, limit, window);
        } else {
          matchNodesOnce<false, false, false>(qG, dG, cur, next, limit, window);
        }
      }
    }

    sizeCur = std::distance(cur->begin(), cur->end());
    sizeNext = std::distance(next->begin(), next->end());
  }

  // match the edges
  galois::do_all(galois::iterate(*cur),
      [&dG, &qG, cur, next] (auto dn) {
        auto& dData = dG.getData(dn);

        for (auto qn: qG) { // multiple matches
          uint64_t mask = (1 << qn);
          if (dData.matched & mask) {
            for (auto qe: qG.edges(qn)) {
              auto qeData = qG.getEdgeData(qe);
              auto qDst = qG.getEdgeDst(qe);

              for (auto de: dG.edges(dn)) {
                auto& deData = dG.getEdgeData(de);
                auto dDst = dG.getEdgeDst(de);
                if (dn < dDst) { // match only one of the symmetric edges
                  if (qeData.label == deData.label) { // TODO: query could be any or multiple labels
                    auto& dDstData = dG.getData(dDst);
                    if (dDstData.matched & (1 << qDst)) {
                      deData.matched |= 1 << *qe;
                    }
                  }
                }
              }
            }
          }
        }
      },
      galois::loopname("MatchNeighborEdges"));
}

template<bool useWindow>
void matchNodeWithRepeatedActionsSelf(Graph &graph, uint32_t nodeLabel, uint32_t action, EventWindow window) {
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (auto n) {
      auto& data = graph.getData(n);
      if (data.label == nodeLabel) {
        unsigned numActions = 0;
        Graph::GraphNode prev = 0;
        for (auto e: graph.edges(n)) {
          auto& eData = graph.getEdgeData(e);
          if (useWindow) {
            if ((eData.timestamp > window.endTime) || (eData.timestamp < window.startTime)) {
              continue; // skip this edge since it is not in the time-span of interest
            }
          }
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
    [&] (auto n) {
      auto& data = graph.getData(n);
      if (data.matched & 1) {
        for (auto e: graph.edges(n)) {
          auto& eData = graph.getEdgeData(e);
          if (useWindow) {
            if ((eData.timestamp > window.endTime) || (eData.timestamp < window.startTime)) {
              continue; // skip this edge since it is not in the time-span of interest
            }
          }
          if (eData.label == action) {
            eData.matched = 1;
            auto dst = graph.getEdgeDst(e);
            auto& dstData = graph.getData(dst);
            dstData.matched |= 2; // atomicity not required
          }
        }
      }
    },
    galois::loopname("MatchNodesDsts"));
}

void matchNodeWithRepeatedActions(Graph &graph, uint32_t nodeLabel, uint32_t action, EventWindow window) {
  // initialize matched
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (auto n) {
      auto& data = graph.getData(n);
      data.matched = 0; // matches to none
    },
    galois::loopname("InitMatched"));

  // match nodes
  if (window.valid) {
    matchNodeWithRepeatedActionsSelf<true>(graph, nodeLabel, action, window);
  } else {
    matchNodeWithRepeatedActionsSelf<false>(graph, nodeLabel, action, window);
  }
}

template<bool useWindow>
void matchNodeWithTwoActionsSelf(Graph &graph, uint32_t nodeLabel, uint32_t action1, uint32_t dstNodeLabel1, uint32_t action2, uint32_t dstNodeLabel2, EventWindow window) {
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (auto n) {
      auto& data = graph.getData(n);
      if (data.label == nodeLabel) {
        bool foundAction1 = false;
        bool foundAction2 = false;
        for (auto e: graph.edges(n)) {
          auto& eData = graph.getEdgeData(e);
          if (useWindow) {
            if ((eData.timestamp > window.endTime) || (eData.timestamp < window.startTime)) {
              continue; // skip this edge since it is not in the time-span of interest
            }
          }
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
    [&] (auto n) {
      auto& data = graph.getData(n);
      if (data.matched & 1) {
        for (auto e: graph.edges(n)) {
          auto& eData = graph.getEdgeData(e);
          if (useWindow) {
            if ((eData.timestamp > window.endTime) || (eData.timestamp < window.startTime)) {
              continue; // skip this edge since it is not in the time-span of interest
            }
          }
          bool mayAction1 = (eData.label == action1);
          bool mayAction2 = (eData.label == action2);
          if (mayAction1 || mayAction2) {
            auto dst = graph.getEdgeDst(e);
            auto& dstData = graph.getData(dst);
            if (mayAction1 && (dstData.label == dstNodeLabel1)) {
              eData.matched = 1;
              dstData.matched |= 2; // atomicity not required
            } else if (mayAction2 && (dstData.label == dstNodeLabel2)) {
              eData.matched = 1;
              dstData.matched |= 4; // atomicity not required
            }
          }
        }
      }
    },
    galois::loopname("MatchNodesDsts"));
}

void matchNodeWithTwoActions(Graph &graph, uint32_t nodeLabel, uint32_t action1, uint32_t dstNodeLabel1, uint32_t action2, uint32_t dstNodeLabel2, EventWindow window) {
  // initialize matched
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (auto n) {
      auto& data = graph.getData(n);
      data.matched = 0; // matches to none
    },
    galois::loopname("InitMatched"));

  // match nodes
  if (window.valid) {
    matchNodeWithTwoActionsSelf<true>(graph, nodeLabel, action1, dstNodeLabel1, action2, dstNodeLabel2, window);
  } else {
    matchNodeWithTwoActionsSelf<false>(graph, nodeLabel, action1, dstNodeLabel1, action2, dstNodeLabel2, window);
  }
}

template<bool useWindow>
void matchNeighborsDsts(Graph& graph, Graph::GraphNode node, uint32_t nodeLabel, uint32_t action, uint32_t neighborLabel, EventWindow window) {
  galois::do_all(galois::iterate(graph.edges(node).begin(), graph.edges(node).end()),
    [&] (auto e) {
      auto& eData = graph.getEdgeData(e);
      if (!useWindow || ((eData.timestamp <= window.endTime) && (eData.timestamp >= window.startTime))) {
        if (eData.label == action) {
          eData.matched = 1;
          auto dst = graph.getEdgeDst(e);
          auto& dstData = graph.getData(dst);
          if (dstData.label == neighborLabel) {
            dstData.matched |= 1; // atomicity not required
          }
        }
      } else {
        // skip this edge since it is not in the time-span of interest
      }
    },
    galois::loopname("MatchNodesDsts"));
}

void matchNeighbors(Graph& graph, Graph::GraphNode node, uint32_t nodeLabel, uint32_t action, uint32_t neighborLabel, EventWindow window) {
  // initialize matched
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (auto n) {
      auto& data = graph.getData(n);
      data.matched = 0; // matches to none
    },
    galois::loopname("InitMatched"));

  // match destinations of node
  assert(graph.getData(node).label == nodeLabel);
  if (window.valid) {
    matchNeighborsDsts<true>(graph, node, nodeLabel, action, neighborLabel, window);
  } else {
    matchNeighborsDsts<false>(graph, node, nodeLabel, action, neighborLabel, window);
  }
}

size_t countMatchedNodes(Graph& graph) {
  galois::GAccumulator<size_t> numMatched;
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (auto n) {
      auto& data = graph.getData(n);
      if (data.matched) {
        numMatched += 1;
      }
    },
    galois::loopname("CountMatchedNodes"));
  return numMatched.reduce();
}

size_t countMatchedNeighbors(Graph& graph, Graph::GraphNode node) {
  galois::GAccumulator<size_t> numMatched;
  // do not count the same node twice (multiple edges to the same node)
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (auto n) {
      auto& data = graph.getData(n);
      if (data.matched) {
        numMatched += 1;
      }
    },
    galois::loopname("CountMatchedNeighbors"));
  return numMatched.reduce();
}

size_t countMatchedEdges(Graph& graph) {
  galois::GAccumulator<size_t> numMatched;
  galois::do_all(galois::iterate(graph.begin(), graph.end()),
    [&] (auto n) {
      auto& data = graph.getData(n);
      if (data.matched) {
        for (auto e: graph.edges(n)) {
          auto eData = graph.getEdgeData(e);
          if (eData.matched) {
            numMatched += 1;
          }
        }
      }
    },
    galois::loopname("CountMatchedEdges"));
  return numMatched.reduce();
}

size_t countMatchedNeighborEdges(Graph& graph, Graph::GraphNode node) {
  galois::GAccumulator<size_t> numMatched;
  galois::do_all(galois::iterate(graph.edges(node).begin(), graph.edges(node).end()),
    [&] (auto e) {
      auto eData = graph.getEdgeData(e);
      if (eData.matched) {
        numMatched += 1; // count the same neighbor for each edge to it
      }
    },
    galois::loopname("CountMatchedEdges"));
  return numMatched.reduce();
}
