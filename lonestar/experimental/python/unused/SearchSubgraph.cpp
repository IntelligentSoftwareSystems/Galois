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

#include "SearchSubgraph.h"
#include "galois/Galois.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/Graph.h"
#include "galois/graphs/TypeTraits.h"
#include "galois/Reduction.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <atomic>
#include <functional>
#include <type_traits>
#include <set>

static std::atomic<size_t> currentlyFound;
static size_t kFound;

struct NodeMatch {
  GNode nQ;
  GNode nD;
  NodeMatch(const GNode q, const GNode d) : nQ(q), nD(d) {}
  NodeMatch() : nQ(), nD() {}
};

typedef std::vector<NodeMatch> Matching;
typedef galois::InsertBag<Matching> MatchingVector;

// check if all attributes in q show up in d
bool contain(const Attr& d, const Attr& q) {
  auto de = d.end();
  for (auto qi : q) {
    auto di = d.find(qi.first);
    if (di == de) {
      return false;
    }
    // ylu: exact match for now. should implement "superset" here.
    if (di->second != qi.second) {
      return false;
    }
  }
  return true;
}

// track for incoming edges for better frontier expansion
struct VF2Algo {
  std::string name() const { return "VF2"; }

  class FilterCandidatesInternal {
    Graph& gD;
    Graph& gQ;
    galois::GReduceLogicalOR& nodeEmpty;
    FilterCandidatesInternal(Graph& d, Graph& q, galois::GReduceLogicalOR& lor)
        : gD(d), gQ(q), nodeEmpty(lor) {}

  public:
    void operator()(const GNode nQ) const {
      auto& dQ = gQ.getData(nQ);

      for (auto nD : gD) {
        auto& dD = gD.getData(nD);

        if (!contain(dD.attr, dQ.attr))
          continue;

        // self loop for nQ but not for nD
        if (gQ.findEdgeSortedByDst(nQ, nQ) != gQ.edge_end(nQ) &&
            gD.findEdgeSortedByDst(nD, nD) == gD.edge_end(nD))
          continue;

        dQ.vVec.push_back(nD);
      }

      std::sort(dQ.vVec.begin(), dQ.vVec.end());
      assert(std::adjacent_find(dQ.vVec.begin(), dQ.vVec.end()) ==
             dQ.vVec.end()); // no duplicates

      nodeEmpty.update(dQ.vVec.empty());
    }

    // return true if at least one node has an empty set of candidates
    static bool go(Graph& gD, Graph& gQ) {
      galois::GReduceLogicalOR isSomeNodeEmpty;
      galois::do_all(gQ, FilterCandidatesInternal(gD, gQ, isSomeNodeEmpty),
                     galois::loopname("filter"), galois::steal());
      return isSomeNodeEmpty.reduce();
    }
  };

  struct SubgraphSearchInternal {
    Graph& gD;
    Graph& gQ;
    MatchingVector& report;
    SubgraphSearchInternal(Graph& d, Graph& q, MatchingVector& r)
        : gD(d), gQ(q), report(r) {}

    struct LocalState {
      template <typename T>
      using PerIterAlloc = typename galois::PerIterAllocTy::rebind<T>::other;

      // query state
      std::set<GNode, std::less<GNode>, PerIterAlloc<GNode>> qFrontier;
      std::set<GNode, std::less<GNode>, PerIterAlloc<GNode>> qMatched;

      // data state
      std::set<GNode, std::less<GNode>, PerIterAlloc<GNode>> dFrontier;
      std::set<GNode, std::less<GNode>, PerIterAlloc<GNode>> dMatched;

      LocalState(galois::PerIterAllocTy& a)
          : qFrontier(a), qMatched(a), dFrontier(a), dMatched(a) {}

      GNode nextQueryNode(Graph& gQ, Matching& matching) {
        if (qFrontier.size())
          return *(qFrontier.begin());
        else
          for (auto nQ : gQ) {
            bool isMatched = false;
            for (auto& mi : matching)
              if (nQ == mi.nQ) {
                isMatched = true;
                break;
              }
            if (!isMatched)
              return nQ;
          }

        // never reaches here. if so, abort.
        abort();
      }
    };

    // for counting occurences only. no space allocation is required.
    template <typename T>
    class counter : public std::iterator<std::output_iterator_tag, T> {
      T dummy;
      long int num;

    public:
      counter() : num(0) {}
      counter& operator++() {
        ++num;
        return *this;
      }
      counter operator++(int) {
        auto retval = *this;
        ++num;
        return retval;
      }
      T& operator*() { return dummy; }
      long int get() { return num; }
    };

    template <typename Graph, typename Set>
    long int countNeighbors(Graph& g, typename Graph::GraphNode n,
                            Set& sMatched) {
      using Iter = typename Graph::edge_iterator;

      auto l = [&g](Iter i) { return g.getEdgeDst(i); };
      counter<typename Graph::GraphNode> count;

      // galois::NoDerefIterator lets dereference return the wrapped iterator
      // itself boost::make_transform_iterator gives an iterator, which is
      // dereferenced to func(in_iter)
      std::set_difference(
          boost::make_transform_iterator(
              galois::NoDerefIterator<Iter>(g.edge_begin(n)), l),
          boost::make_transform_iterator(
              galois::NoDerefIterator<Iter>(g.edge_end(n)), l),
          sMatched.begin(), sMatched.end(), count);
      return count.get();
    }

#if DIRECTED
    template <typename Graph, typename Set>
    long int countInNeighbors(Graph& g, typename Graph::GraphNode n,
                              Set& sMatched) {
      using Iter = typename Graph::in_edge_iterator;

      // lambda expression. captures Graph& for expression body.
      auto l = [&g](Iter i) { return g.getEdgeDst(i); };

      counter<typename Graph::GraphNode> count;

      std::set_difference(
          boost::make_transform_iterator(
              galois::NoDerefIterator<Iter>(g.in_edge_begin(n)), l),
          boost::make_transform_iterator(
              galois::NoDerefIterator<Iter>(g.in_edge_end(n)), l),
          sMatched.begin(), sMatched.end(), count);
      return count.get();
    }
#endif // DIRECTED

    std::vector<GNode, LocalState::PerIterAlloc<GNode>>
    refineCandidates(Graph& gD, Graph& gQ, GNode nQuery,
                     galois::PerIterAllocTy& alloc, LocalState& state) {
      std::vector<GNode, LocalState::PerIterAlloc<GNode>> refined(alloc);
      auto numNghQ = std::distance(gQ.edge_begin(nQuery), gQ.edge_end(nQuery));
      long int numUnmatchedNghQ = countNeighbors(gQ, nQuery, state.qMatched);

#if DIRECTED
      long int numInNghQ = 0, numUnmatchedInNghQ = 0;
      numInNghQ =
          std::distance(gQ.in_edge_begin(nQuery), gQ.in_edge_end(nQuery));
      numUnmatchedInNghQ = countInNeighbors(gQ, nQuery, state.qMatched);
#endif // DIRECTED

      // consider all nodes in data frontier
      auto& dQ = gQ.getData(nQuery);
      for (auto ii : state.dFrontier) {
        // not a candidate for nQuery
        if (!std::binary_search(dQ.vVec.begin(), dQ.vVec.end(), ii))
          continue;

        auto numNghD = std::distance(gD.edge_begin(ii), gD.edge_end(ii));
        if (numNghD < numNghQ)
          continue;

        long int numUnmatchedNghD = countNeighbors(gD, ii, state.dMatched);
        if (numUnmatchedNghD < numUnmatchedNghQ)
          continue;

#if DIRECTED
        auto numInNghD =
            std::distance(gD.in_edge_begin(ii), gD.in_edge_end(ii));
        if (numInNghD < numInNghQ)
          continue;

        long int numUnmatchedInNghD = countInNeighbors(gD, ii, state.dMatched);
        if (numUnmatchedInNghD < numUnmatchedInNghQ)
          continue;
#endif // DIRECTED

        refined.push_back(ii);
      }
      return refined;
    }

    bool isJoinable(Graph& gD, Graph& gQ, GNode nD, GNode nQ,
                    Matching& matching) {
      for (auto& nm : matching) {
        // nD is already matched
        if (nD == nm.nD)
          return false;

        // nQ => (nm.nQ) exists but not nD => (nm.nD)
        if (gQ.findEdgeSortedByDst(nQ, nm.nQ) != gQ.edge_end(nQ) &&
            gD.findEdgeSortedByDst(nD, nm.nD) == gD.edge_end(nD))
          return false;

#if DIRECTED
        // (nm.nQ) => nQ exists but not (nm.nD) => nD
        if (gQ.findEdgeSortedByDst(nm.nQ, nQ) != gQ.edge_end(nm.nQ) &&
            gD.findEdgeSortedByDst(nm.nD, nD) == gD.edge_end(nm.nD))
          return false;
#endif // DIRECTED
      }

      return true;
    }

    template <typename StateSet, typename StepSet, typename Graph>
    void insertDstFrontierTracked(StateSet& sMatched, StateSet& sFrontier,
                                  StepSet& sAdd2F, Graph& g,
                                  typename Graph::GraphNode n) {
      for (auto e : g.edges(n)) {
        auto ngh = g.getEdgeDst(e);
        if (!sMatched.count(ngh) && sFrontier.insert(ngh).second)
          sAdd2F.push_back(ngh);
      }
    }

#if DIRECTED
    template <typename StateSet, typename StepSet, typename Graph>
    void insertInDstFrontierTracked(StateSet& sMatched, StateSet& sFrontier,
                                    StepSet& sAdd2F, Graph& g,
                                    typename Graph::GraphNode n) {
      for (auto ie : g.in_edges(n)) {
        auto ngh = g.getEdgeDst(ie);
        if (!sMatched.count(ngh) && sFrontier.insert(ngh).second)
          sAdd2F.push_back(ngh);
      }
    }
#endif // DIRECTED

    void doSearch(LocalState& state, Matching& matching,
                  galois::PerIterAllocTy& alloc) {
      if (currentlyFound.load() >= kFound)
        return;

      if (matching.size() == gQ.size()) {
        report.push_back(matching);
        currentlyFound += 1;
        return;
      }

      auto nQ      = state.nextQueryNode(gQ, matching);
      auto refined = refineCandidates(gD, gQ, nQ, alloc, state);

      // update query state
      state.qMatched.insert(nQ);
      state.qFrontier.erase(nQ);

      std::vector<GNode, LocalState::PerIterAlloc<GNode>> qAdd2Frontier(alloc);
      insertDstFrontierTracked(state.qMatched, state.qFrontier, qAdd2Frontier,
                               gQ, nQ);
#if DIRECTED
      insertInDstFrontierTracked(state.qMatched, state.qFrontier, qAdd2Frontier,
                                 gQ, nQ);
#endif // DIRECTED

      // search for all possible candidate data nodes
      for (auto r : refined) {
        if (!isJoinable(gD, gQ, r, nQ, matching))
          continue;

        // add (nQ, r) to matching
        matching.push_back(NodeMatch(nQ, r));

        // update data state
        state.dMatched.insert(r);
        state.dFrontier.erase(r);

        std::vector<GNode, LocalState::PerIterAlloc<GNode>> dAdd2Frontier(
            alloc);
        insertDstFrontierTracked(state.dMatched, state.dFrontier, dAdd2Frontier,
                                 gD, r);
#if DIRECTED
        insertInDstFrontierTracked(state.dMatched, state.dFrontier,
                                   dAdd2Frontier, gD, r);
#endif // DIRECTED

        doSearch(state, matching, alloc);
        if (currentlyFound.load() >= kFound)
          return;

        // restore data state
        state.dMatched.erase(r);
        state.dFrontier.insert(r);
        for (auto i : dAdd2Frontier)
          state.dFrontier.erase(i);
        dAdd2Frontier.clear();

        // remove (nQ, r) from matching
        matching.pop_back();
      }

      // restore query state
      state.qMatched.erase(nQ);
      state.qFrontier.insert(nQ);
      for (auto i : qAdd2Frontier)
        state.qFrontier.erase(i);
    }

    template <typename Set, typename Graph>
    void insertDstFrontier(Set& sMatched, Set& sFrontier, Graph& g,
                           typename Graph::GraphNode n) {
      for (auto e : g.edges(n)) {
        auto ngh = g.getEdgeDst(e);
        if (!sMatched.count(ngh))
          sFrontier.insert(ngh);
      }
    }

#if DIRECTED
    template <typename Set, typename Graph>
    void insertInDstFrontier(Set& sMatched, Set& sFrontier, Graph& g,
                             typename Graph::GraphNode n) {
      for (auto ie : g.in_edges(n)) {
        auto ngh = g.getEdgeDst(ie);
        if (!sMatched.count(ngh))
          sFrontier.insert(ngh);
      }
    }
#endif // DIRECTED

    // galois::for_each expects ctx
    void operator()(NodeMatch& seed, galois::UserContext<NodeMatch>& ctx) {
      LocalState state(ctx.getPerIterAlloc());

      auto nQ = seed.nQ;
      state.qMatched.insert(nQ);

      insertDstFrontier(state.qMatched, state.qFrontier, gQ, nQ);
#if DIRECTED
      insertInDstFrontier(state.qMatched, state.qFrontier, gQ, nQ);
#endif // DIRECTED

      auto nD = seed.nD;
      state.dMatched.insert(nD);

      insertDstFrontier(state.dMatched, state.dFrontier, gD, nD);
#if DIRECTED
      insertInDstFrontier(state.dMatched, state.dFrontier, gD, nD);
#endif // DIRECTED

      Matching matching{seed};
      doSearch(state, matching, ctx.getPerIterAlloc());

      if (currentlyFound.load() >= kFound)
        ctx.breakLoop();
    }
  };

public:
  // return true if at least one node has an empty set of candidates
  static bool filterCandidates(Graph& gD, Graph& gQ) {
    return FilterCandidatesInternal::go(gD, gQ);
  }

  static MatchingVector subgraphSearch(Graph& gD, Graph& gQ) {
    // parallelize the search for candidates of gQ.begin()
    galois::InsertBag<NodeMatch> works;
    auto nQ = *(gQ.begin());
    for (auto c : gQ.getData(nQ).vVec)
      works.push_back(NodeMatch(nQ, c));

    MatchingVector report;
    galois::for_each(works, SubgraphSearchInternal(gD, gQ, report),
                     galois::loopname("search_for_each"),
                     galois::no_conflicts(), galois::no_pushes(),
                     galois::parallel_break(), galois::per_iter_alloc());
    return report;
  }
};

struct UllmannAlgo {
  std::string name() const { return "Ullmann"; }

  struct FilterCandidatesInternal {
    Graph& gD;
    Graph& gQ;
    galois::GReduceLogicalOR& nodeEmpty;
    FilterCandidatesInternal(Graph& d, Graph& q, galois::GReduceLogicalOR& lor)
        : gD(d), gQ(q), nodeEmpty(lor) {}

    void operator()(const GNode nQ) const {
      auto& dQ = gQ.getData(nQ);

      for (auto nD : gD) {
        auto& dD = gD.getData(nD);

        if (!contain(dD.attr, dQ.attr))
          continue;

        // self loop for nQ but not for nD
        if (gQ.findEdgeSortedByDst(nQ, nQ) != gQ.edge_end(nQ) &&
            gD.findEdgeSortedByDst(nD, nD) == gD.edge_end(nD))
          continue;

        dQ.vVec.push_back(nD);
      }

      nodeEmpty.update(dQ.vVec.empty());
    }

    // return true if at least one node has an empty set of candidates
    static bool go(Graph& gD, Graph& gQ) {
      galois::GReduceLogicalOR isSomeNodeEmpty;
      galois::do_all(gQ, FilterCandidatesInternal(gD, gQ, isSomeNodeEmpty),
                     galois::loopname("filter"), galois::steal());
      return isSomeNodeEmpty.reduce();
    }
  };

  struct SubgraphSearchInternal {
    Graph& gD;
    Graph& gQ;
    MatchingVector& report;
    SubgraphSearchInternal(Graph& d, Graph& q, MatchingVector& r)
        : gD(d), gQ(q), report(r) {}

    GNode nextQueryNode(Graph& gQ, Matching& matching) {
      auto qi = gQ.begin();
      std::advance(qi, matching.size());
      return *qi;
    }

    std::vector<GNode> refineCandidates(Graph& gD, Graph& gQ, GNode nQuery,
                                        Matching& matching) {
      std::vector<GNode> refined;
      auto& dQ     = gQ.getData(nQuery);
      auto numNghQ = std::distance(gQ.edge_begin(nQuery), gQ.edge_end(nQuery));
#if DIRECTED
      auto numInNghQ =
          std::distance(gQ.in_edge_begin(nQuery), gQ.in_edge_end(nQuery));
#endif // DIRECTED

      for (auto c : dQ.vVec) {
        auto numNghD = std::distance(gD.edge_begin(c), gD.edge_end(c));
#if DIRECTED
        auto numInNghD = std::distance(gD.in_edge_begin(c), gD.in_edge_end(c));
#endif // DIRECTED

        if (numNghD >= numNghQ)
#if DIRECTED
          if (numInNghD >= numInNghQ)
#endif // DIRECTED
            refined.push_back(c);
      }

      return refined;
    }

    bool isJoinable(Graph& gD, Graph& gQ, GNode nD, GNode nQ,
                    Matching& matching) {
      for (auto& nm : matching) {
        // nD is already matched
        if (nD == nm.nD)
          return false;

        // nQ => (nm.nQ) exists but not nD => (nm.nD)
        if (gQ.findEdgeSortedByDst(nQ, nm.nQ) != gQ.edge_end(nQ) &&
            gD.findEdgeSortedByDst(nD, nm.nD) == gD.edge_end(nD))
          return false;

        // (nm.nQ) => nQ exists but not (nm.nD) => nD
        if (gQ.findEdgeSortedByDst(nm.nQ, nQ) != gQ.edge_end(nm.nQ) &&
            gD.findEdgeSortedByDst(nm.nD, nD) == gD.edge_end(nm.nD))
          return false;
      }

      return true;
    }

    void doSearch(Matching& matching) {
      if (currentlyFound.load() >= kFound)
        return;

      if (matching.size() == gQ.size()) {
        report.push_back(matching);
        currentlyFound += 1;
        return;
      }

      auto nQ      = nextQueryNode(gQ, matching);
      auto refined = refineCandidates(gD, gQ, nQ, matching);

      for (auto r : refined) {
        if (!isJoinable(gD, gQ, r, nQ, matching))
          continue;

        // add (nQ, r) to matching
        matching.push_back(NodeMatch(nQ, r));

        doSearch(matching);
        if (currentlyFound.load() >= kFound)
          return;

        // remove (nQ, r) from matching
        matching.pop_back();
      }
    }

    // galois::for_each expects ctx
    void operator()(NodeMatch& seed, galois::UserContext<NodeMatch>& ctx) {
      Matching matching{seed};
      doSearch(matching);
      if (currentlyFound.load() >= kFound)
        ctx.breakLoop();
    }
  };

public:
  // return true if at least one node has an empty set of candidates
  static bool filterCandidates(Graph& gD, Graph& gQ) {
    return FilterCandidatesInternal::go(gD, gQ);
  }

  static MatchingVector subgraphSearch(Graph& gD, Graph& gQ) {
    // parallelize the search for candidates of gQ.begin()
    galois::InsertBag<NodeMatch> works;
    auto nQ = *(gQ.begin());
    for (auto c : gQ.getData(nQ).vVec)
      works.push_back(NodeMatch(nQ, c));

    MatchingVector report;
    galois::for_each(works, SubgraphSearchInternal(gD, gQ, report),
                     galois::loopname("search_for_each"),
                     galois::no_conflicts(), galois::no_pushes(),
                     galois::parallel_break());
    return report;
  }
};

// check if the matching is correct
void verifyMatching(Matching& matching, Graph& gD, Graph& gQ) {
  bool isFailed = false;

  for (auto& nm1 : matching) {
    auto& dQ1 = gQ.getData(nm1.nQ);
    auto& dD1 = gD.getData(nm1.nD);

    if (!contain(dD1.attr, dQ1.attr)) {
      isFailed = true;
      std::cerr << "attr not match: gQ(" << nm1.nQ << ") not contained by gD("
                << nm1.nD << ")" << std::endl;
    }

    for (auto& nm2 : matching) {
      // two distinct query nodes map to the same data node
      if (nm1.nQ != nm2.nQ && nm1.nD == nm2.nD) {
        isFailed = true;
        std::cerr << "inconsistent mapping to data node: gQ(" << nm1.nQ;
        std::cerr << ") to gD(" << nm1.nD << "), gQ(" << nm2.nQ;
        std::cerr << ") to gD(" << nm2.nD << ")" << std::endl;
      }

      // a query node mapped to different data nodes
      if (nm1.nQ == nm2.nQ && nm1.nD != nm2.nD) {
        isFailed = true;
        std::cerr << "inconsistent mapping from query node: gQ(" << nm1.nQ;
        std::cerr << ") to gD(" << nm1.nD << "), gQ(" << nm2.nQ;
        std::cerr << ") to gD(" << nm2.nD << ")" << std::endl;
      }

      // query edge not matched to data edge
      if (gQ.findEdgeSortedByDst(nm1.nQ, nm2.nQ) != gQ.edge_end(nm1.nQ) &&
          gD.findEdgeSortedByDst(nm1.nD, nm2.nD) == gD.edge_end(nm1.nD)) {
        isFailed = true;
        std::cerr << "edge not match: gQ(" << nm1.nQ << " => " << nm2.nQ;
        std::cerr << "), but no gD(" << nm1.nD << " => " << nm2.nD << ")"
                  << std::endl;
      }
    }
  }

  if (isFailed)
    GALOIS_DIE("Verification failed");
}

NodePair* reportMatchings(MatchingVector& report, size_t size) {
  NodePair* result = new NodePair[size]();
  size_t i         = 0;
  for (auto& m : report) {
    for (auto& nm : m) {
      result[i].nQ   = nm.nQ;
      result[i++].nD = nm.nD;
      if (i == size)
        break;
    }
    if (i == size)
      break;
  }

#if 0
  size_t gQSize = size / kFound;
  for(auto j = 0; j < kFound; ++j) {
    std::cout << "GraphMatch " << j << std::endl;
    for(auto k = 0; k < gQSize; ++k) {
      std::cout << "  gQ(" << result[j*gQSize+k].nQ << ") -> gD(" << result[j*gQSize+k].nD << ")" << std::endl;
    }
    std::cout << std::endl;
  }
#endif

  return result;
}

void constructNodeVec(Graph& gQ) {
  using vector_type = std::vector<GNode>;
  galois::do_all(
      gQ,
      [&gQ](const GNode n) {
        // placement new
        new (&(gQ.getData(n).vVec)) vector_type();
      },
      galois::steal());
}

void destructNodeVec(Graph& gQ) {
  using vector_type = std::vector<GNode>;
  galois::do_all(
      gQ, [&gQ](const GNode n) { gQ.getData(n).vVec.~vector_type(); },
      galois::steal());
}

template <typename Algo>
NodePair* run(Graph& gD, Graph& gQ) {
  Algo algo;
  //  std::cout << "Running " << algo.name() << " Algorithm..." << std::endl;

  gD.sortAllEdgesByDst();
  gQ.sortAllEdgesByDst();
  constructNodeVec(gQ);

  //  galois::StatTimer T;
  //  T.start();

  //  galois::StatTimer filterT("FilterCandidates");
  //  filterT.start();
  bool isSomeNodeUnmatched = algo.filterCandidates(gD, gQ);
  //  filterT.stop();

  if (isSomeNodeUnmatched) {
    //    T.stop();
    std::cout << "Some nodes have no candidates to match." << std::endl;
    destructNodeVec(gQ);
    return (new NodePair[gQ.size() * kFound]());
  }

  //  galois::StatTimer searchT("SubgraphSearch");
  //  searchT.start();
  currentlyFound.store(0);
  MatchingVector report = algo.subgraphSearch(gD, gQ);
  //  searchT.stop();

  //  T.stop();
  //  std::cout << "Found " << currentlyFound << " instance(s) of the query
  //  graph." << std::endl;
  if (currentlyFound) {
    for (auto& m : report)
      verifyMatching(m, gD, gQ);
    //    std::cout << "Verification succeeded" << std::endl;
  }
  destructNodeVec(gQ);
  return reportMatchings(report, gQ.size() * kFound);
}

NodePair* searchSubgraphUllmann(Graph* gD, Graph* gQ, size_t k) {
  //  galois::StatManager statManager;

  kFound = k;

  //  galois::StatTimer T("TotalTime");
  //  T.start();
  NodePair* result = run<UllmannAlgo>(*gD, *gQ);
  //  T.stop();

  return result;
}

NodePair* searchSubgraphVF2(Graph* gD, Graph* gQ, size_t k) {
  //  galois::StatManager statManager;

  kFound = k;

  //  galois::StatTimer T("TotalTime");
  //  T.start();
  NodePair* result = run<VF2Algo>(*gD, *gQ);
  //  T.stop();

  return result;
}
