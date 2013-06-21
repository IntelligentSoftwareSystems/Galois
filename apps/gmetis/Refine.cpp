/** GMetis -*- C++ -*-
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
 * @author Xin Sui <xinsui@cs.utexas.edu>
 * @author Nikunj Yadav <nikunj@cs.utexas.edu>
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#include "Metis.h"

namespace {

struct gainIndexer {
  static GGraph* g;

  int operator()(GNode n) {
    int retval = 0;
    Galois::MethodFlag flag = Galois::NONE;
    unsigned nPart = g->getData(n, flag).getPart();
    for (auto ii = g->edge_begin(n, flag), ee = g->edge_end(n); ii != ee; ++ii) {
      GNode neigh = g->getEdgeDst(ii, flag);
      if (g->getData(neigh, flag).getPart() == nPart)
        retval -= g->getEdgeData(ii, flag);
      else
        retval += g->getEdgeData(ii, flag);
    }
    return -retval / 16;
  }
};

GGraph* gainIndexer::g;

bool isBoundary(GGraph& g, GNode n) {
  unsigned nPart = g.getData(n).getPart();
  for (auto ii = g.edge_begin(n), ee =g.edge_end(n); ii != ee; ++ii)
    if (g.getData(g.getEdgeDst(ii)).getPart() != nPart)
      return true;
  return false;
}

struct findBoundary {
  Galois::InsertBag<GNode>& b;
  GGraph& g;
  findBoundary(Galois::InsertBag<GNode>& _b, GGraph& _g) :b(_b), g(_g) {}
  void operator()(GNode n) {
    if (isBoundary(g, n))
      b.push(n);
  }
};

template<bool ignoreSizeOnSelf>
struct refine_BKL2 {
  unsigned maxSize;
  GGraph& g;
  std::vector<partInfo>& parts;

  refine_BKL2(unsigned ms, GGraph& _g, std::vector<partInfo>& _p) : maxSize(ms), g(_g), parts(_p) {}

  //Find the partition n is most connected to
  unsigned pickPartition(GNode n) {
    std::vector<unsigned> edges(parts.size(), 0);
    unsigned P = g.getData(n).getPart();
    for (auto ii = g.edge_begin(n), ee =g.edge_end(n); ii != ee; ++ii) {
      GNode neigh = g.getEdgeDst(ii);
      auto& nd = g.getData(neigh);
      if (parts[nd.getPart()].partWeight < maxSize
          || (ignoreSizeOnSelf && nd.getPart() == P))
        edges[nd.getPart()] += g.getEdgeData(ii);
    }
    return std::distance(edges.begin(), std::max_element(edges.begin(), edges.end()));
  }

  template<typename Context>
  void operator()(GNode n, Context& cnx) {
    auto& nd = g.getData(n);
    unsigned curpart = nd.getPart();
    unsigned newpart = pickPartition(n);
    if (curpart != newpart) {
      nd.setPart(newpart);
      //__sync_fetch_and_sub(&maxSize, 1);
      __sync_fetch_and_sub(&parts[curpart].partWeight, nd.getWeight());
      __sync_fetch_and_add(&parts[newpart].partWeight, nd.getWeight());
      __sync_fetch_and_sub(&parts[curpart].partSize, 1);
      __sync_fetch_and_add(&parts[newpart].partSize, 1);
      for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii != ee; ++ii) {
        GNode neigh = g.getEdgeDst(ii);
        auto& ned = g.getData(neigh);
        if (ned.getPart() != newpart)
          cnx.push(neigh);
      }
    }
  }

  static void go(unsigned ms, GGraph& gg, std::vector<partInfo>& p) {
    typedef Galois::WorkList::dChunkedFIFO<8> Chunk;
    typedef Galois::WorkList::OrderedByIntegerMetric<gainIndexer, Chunk, 10> pG;
    gainIndexer::g = &gg;
    Galois::InsertBag<GNode> boundary;
    Galois::do_all_local(gg, findBoundary(boundary, gg), "boundary");
    Galois::for_each_local<pG>(boundary, refine_BKL2(ms, gg, p), "refine");
  }
};

struct projectPart {
  GGraph* fineGraph;
  GGraph* coarseGraph;
  std::vector<partInfo>& parts;

  projectPart(MetisGraph* Graph, std::vector<partInfo>& p) :fineGraph(Graph->getFinerGraph()->getGraph()), coarseGraph(Graph->getGraph()), parts(p) {}

  void operator()(GNode n) {
    auto& cn = coarseGraph->getData(n);
    unsigned part = cn.getPart();
    for (unsigned x = 0; x < cn.numChildren(); ++x)
      fineGraph->getData(cn.getChild(x)).setPart(part);
    //This slows us down.  I don't think we need size (number of nodes in the current coarsening level)
    if (cn.numChildren() > 1)
      __sync_fetch_and_add(&parts[part].partSize, cn.numChildren() - 1);
  }

  static void go(MetisGraph* Graph, std::vector<partInfo>& p) {
    Galois::do_all_local(*Graph->getGraph(), projectPart(Graph, p), "project");
  }
};

} //anon namespace

void refine(MetisGraph* coarseGraph, std::vector<partInfo>& parts, unsigned maxSize) {
  do {
    //refine nparts times
    refine_BKL2<true>::go(maxSize, *coarseGraph->getGraph(), parts);
    // std::cout << "Refinement of " << coarseGraph->getGraph() << "\n";
    // printPartStats(parts);

    //project up
    MetisGraph* fineGraph = coarseGraph->getFinerGraph();
    if (fineGraph) {
      projectPart::go(coarseGraph, parts);
    }
  } while ((coarseGraph = coarseGraph->getFinerGraph()));
}

void balance(MetisGraph* coarseGraph, std::vector<partInfo>& parts, unsigned maxSize) {
  refine_BKL2<false>::go(maxSize, *coarseGraph->getGraph(), parts);
}

