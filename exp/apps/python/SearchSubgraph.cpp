#include "SearchSubgraph.h"
#include "Galois/Galois.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/Timer.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Graphs/TypeTraits.h"
#include "Galois/Accumulator.h"

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
  NodeMatch(const GNode q, const GNode d): nQ(q), nD(d) {}
  NodeMatch(): nQ(), nD() {}
};

typedef std::vector<NodeMatch> Matching;
typedef Galois::InsertBag<Matching> MatchingVector;

// check if all attributes in q show up in d
bool contain(const Attr& d, const Attr& q) {
  auto de = d.end();
  for(auto qi: q) {
    auto di = d.find(qi.first);
    if(di == de) {
      return false;
    }
    // ylu: exact match for now. should implement "superset" here.
    if(di->second != qi.second) {
      return false;
    }
  }
  return true;
}

struct UllmannAlgo {
  std::string name() const { return "Ullmann"; }

  struct FilterCandidatesInternal {
    Graph& gD;
    Graph& gQ;
    Galois::GReduceLogicalOR& nodeEmpty;
    FilterCandidatesInternal(Graph& d, Graph& q, Galois::GReduceLogicalOR& lor): gD(d), gQ(q), nodeEmpty(lor) {}

    void operator()(const GNode nQ) const {
      auto& dQ = gQ.getData(nQ);

      dQ.mode = 3; // set to use vVec
      dQ.vVec.clear();

      for(auto nD : gD) {
        auto& dD = gD.getData(nD);

        if(!contain(dD.attr, dQ.attr))
          continue;

        // self loop for nQ but not for nD
        if(gQ.findEdgeSortedByDst(nQ, nQ) != gQ.edge_end(nQ) && gD.findEdgeSortedByDst(nD, nD) == gD.edge_end(nD))
          continue;

        dQ.vVec.push_back(nD);
      }

      nodeEmpty.update(dQ.vVec.empty());
    }

    // return true if at least one node has an empty set of candidates
    static bool go(Graph& gD, Graph& gQ) {
      Galois::GReduceLogicalOR isSomeNodeEmpty;
      Galois::do_all_local(gQ, FilterCandidatesInternal(gD, gQ, isSomeNodeEmpty), Galois::loopname("filter"), Galois::do_all_steal<true>());
      return isSomeNodeEmpty.reduce();
    }
  };

  struct SubgraphSearchInternal {
    Graph& gD;
    Graph& gQ;
    MatchingVector& report;
    SubgraphSearchInternal(Graph& d, Graph& q, MatchingVector& r): gD(d), gQ(q), report(r) {}

    GNode nextQueryNode(Graph& gQ, Matching& matching) {
      auto qi = gQ.begin();
      std::advance(qi, matching.size());
      return *qi;
    }

    std::vector<GNode> refineCandidates(Graph& gD, Graph& gQ, GNode nQuery, Matching& matching) {
      std::vector<GNode> refined;
      auto& dQ = gQ.getData(nQuery);
      auto numNghQ = std::distance(gQ.edge_begin(nQuery), gQ.edge_end(nQuery));
      auto numInNghQ = std::distance(gQ.in_edge_begin(nQuery), gQ.in_edge_end(nQuery));

      for(auto c : dQ.vVec) {
        auto numNghD = std::distance(gD.edge_begin(c), gD.edge_end(c));
        auto numInNghD = std::distance(gD.in_edge_begin(c), gD.in_edge_end(c));
        if(numNghD >= numNghQ && numInNghD >= numInNghQ)
          refined.push_back(c);
      }

      return refined;
    }

    bool isJoinable(Graph& gD, Graph& gQ, GNode nD, GNode nQ, Matching& matching) {
      for(auto& nm : matching) {
        // nD is already matched
        if(nD == nm.nD)
          return false;

        // nQ => (nm.nQ) exists but not nD => (nm.nD)
        if(gQ.findEdgeSortedByDst(nQ, nm.nQ) != gQ.edge_end(nQ) && gD.findEdgeSortedByDst(nD, nm.nD) == gD.edge_end(nD))
          return false;

        // (nm.nQ) => nQ exists but not (nm.nD) => nD
        if(gQ.findEdgeSortedByDst(nm.nQ, nQ) != gQ.edge_end(nm.nQ) && gD.findEdgeSortedByDst(nm.nD, nD) == gD.edge_end(nm.nD))
          return false;
      }

      return true;
    }

    void doSearch(Matching& matching) {
      if(currentlyFound.load() >= kFound)
        return;

      if(matching.size() == gQ.size()) {
        report.push_back(matching);
        currentlyFound += 1;
        return;
      }

      auto nQ = nextQueryNode(gQ, matching);
      auto refined = refineCandidates(gD, gQ, nQ, matching);

      for(auto r : refined) {
        if(!isJoinable(gD, gQ, r, nQ, matching))
          continue;

        // add (nQ, r) to matching 
        matching.push_back(NodeMatch(nQ, r));

        doSearch(matching);
        if(currentlyFound.load() >= kFound)
          return;

        // remove (nQ, r) from matching
        matching.pop_back();
      }
    }

    // Galois::for_each expects ctx
    void operator()(NodeMatch& seed, Galois::UserContext<NodeMatch>& ctx) {
      Matching matching{seed};
      doSearch(matching);
      if(currentlyFound.load() >= kFound)
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
    Galois::InsertBag<NodeMatch> works;
    auto nQ = *(gQ.begin());
    for(auto c : gQ.getData(nQ).vVec)
      works.push_back(NodeMatch(nQ, c));

    MatchingVector report;
    Galois::for_each_local(works, SubgraphSearchInternal(gD, gQ, report), Galois::loopname("search_for_each"), 
      Galois::does_not_need_aborts<>(), Galois::does_not_need_push<>(), Galois::needs_parallel_break<>());
    return report;
  }
};

// check if the matching is correct
void verifyMatching(Matching& matching, Graph& gD, Graph& gQ) {
  bool isFailed = false;

  for(auto& nm1 : matching) {
    auto& dQ1 = gQ.getData(nm1.nQ);
    auto& dD1 = gD.getData(nm1.nD);

    if(!contain(dD1.attr, dQ1.attr)) {
      isFailed = true;
      std::cerr << "attr not match: gQ(" << nm1.nQ << ") not contained by gD(" << nm1.nD << ")" << std::endl;
    }

    for(auto& nm2 : matching) {
      // two distinct query nodes map to the same data node
      if(nm1.nQ != nm2.nQ && nm1.nD == nm2.nD) {
        isFailed = true;
        std::cerr << "inconsistent mapping to data node: gQ(" << nm1.nQ;
        std::cerr << ") to gD(" << nm1.nD << "), gQ(" << nm2.nQ;
        std::cerr << ") to gD(" << nm2.nD << ")" << std::endl;
      }

      // a query node mapped to different data nodes
      if(nm1.nQ == nm2.nQ && nm1.nD != nm2.nD) {
        isFailed = true;
        std::cerr << "inconsistent mapping from query node: gQ(" << nm1.nQ;
        std::cerr << ") to gD(" << nm1.nD << "), gQ(" << nm2.nQ;
        std::cerr << ") to gD(" << nm2.nD << ")" << std::endl;
      }

      // query edge not matched to data edge
      if(gQ.findEdgeSortedByDst(nm1.nQ, nm2.nQ) != gQ.edge_end(nm1.nQ) && gD.findEdgeSortedByDst(nm1.nD, nm2.nD) == gD.edge_end(nm1.nD)) {
        isFailed = true;
        std::cerr << "edge not match: gQ(" << nm1.nQ << " => " << nm2.nQ;
        std::cerr << "), but no gD(" << nm1.nD << " => " << nm2.nD << ")" << std::endl;
      }
    }
  }

  if(isFailed)
    GALOIS_DIE("Verification failed");
}

NodePair *reportMatchings(MatchingVector& report, size_t size) {
  NodePair *result = new NodePair [size] ();
  size_t i = 0;
  for(auto& m : report) {
    for(auto& nm : m) {
      result[i].nQ = nm.nQ;
      result[i++].nD = nm.nD;
      if(i == size) {
        break;
      }
    }
    if(i == size) {
      break;
    }
  } 
  return result;
}

template<typename Algo>
NodePair *run(Graph& gD, Graph& gQ) {
  Algo algo;
  std::cout << "Running " << algo.name() << " Algorithm..." << std::endl;

  gD.sortAllEdgesByDst();
  gQ.sortAllEdgesByDst();

  Galois::StatTimer T;
  T.start();

  Galois::StatTimer filterT("FilterCandidates");
  filterT.start();
  bool isSomeNodeUnmatched = algo.filterCandidates(gD, gQ);
  filterT.stop();

  if(isSomeNodeUnmatched) {
    T.stop();
    std::cout << "Some nodes have no candidates to match." << std::endl;
    return (new NodePair [gQ.size()*kFound] ());
  }

  Galois::StatTimer searchT("SubgraphSearch");
  searchT.start();
  currentlyFound.store(0);
  MatchingVector report = algo.subgraphSearch(gD, gQ);
  searchT.stop();

  T.stop();
  std::cout << "Found " << currentlyFound << " instance(s) of the query graph." << std::endl;
  if(currentlyFound) {
    for(auto& m: report)
      verifyMatching(m, gD, gQ);
    std::cout << "Verification succeeded" << std::endl;
  }
  return reportMatchings(report, gQ.size()*kFound);
}

NodePair *searchSubgraphUllmann(Graph *gD, Graph *gQ, size_t k) {
  Galois::StatManager statManager; 

  kFound = k;

  Galois::StatTimer T("TotalTime");
  T.start();
  NodePair *result = run<UllmannAlgo>(*gD, *gQ);
  T.stop();

  return result;
}
#if 0
NodePair *searchSubgraphVF2(Graph *gD, Graph *gQ, size_t k) {
  Galois::StatManager statManager; 

  kFound = k;

  Galois::StatTimer T("TotalTime");
  T.start();
//  NodePair *result = run<VF2Algo>(*gD, *gQ);
  T.stop();

  return result;
}
#endif

void deleteGraphMatches(NodePair *pairs) {
  delete[] pairs;
}

