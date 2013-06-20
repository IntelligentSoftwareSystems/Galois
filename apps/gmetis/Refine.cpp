#include "GMetisConfig.h"
#include "MetisGraph.h"
#include "Metis.h"

int gain(GGraph& g, GNode n) {
  int retval = 0;
  unsigned nPart = g.getData(n).getPart();
  for (auto ii = g.edge_begin(n), ee =g.edge_end(n); ii != ee; ++ii) {
    GNode neigh = g.getEdgeDst(ii);
    if (g.getData(neigh).getPart() == nPart)
      retval -= g.getEdgeData(ii);
    else
      retval += g.getEdgeData(ii);
  }
  return retval;
}

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
      if (parts[nd.getPart()].partWeight < maxSize || nd.getPart() == P)
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
    Galois::InsertBag<GNode> boundary;
    Galois::do_all_local(gg, findBoundary(boundary, gg));
    Galois::for_each_local(boundary, refine_BKL2(ms, gg, p));
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
      fineGraph->getData(cn.getChild(0)).setPart(part);
    //This slows us down.  I don't think we need size (number of nodes in the current coarsening level)
    if (cn.numChildren() > 1)
      __sync_fetch_and_add(&parts[part].partSize, cn.numChildren() - 1);
  }

  static void go(MetisGraph* Graph, std::vector<partInfo>& p) {
    Galois::do_all_local(*Graph->getGraph(), projectPart(Graph, p), "project");
  }
};

void refine(MetisGraph* coarseGraph, std::vector<partInfo>& parts, unsigned maxSize) {
  do {
    //refine nparts times
    refine_BKL2::go(maxSize, *coarseGraph->getGraph(), parts);
    // std::cout << "Refinement of " << coarseGraph->getGraph() << "\n";
    // printPartStats(parts);

    //project up
    MetisGraph* fineGraph = coarseGraph->getFinerGraph();
    if (fineGraph) {
      projectPart::go(coarseGraph, parts);
    }
  } while ((coarseGraph = coarseGraph->getFinerGraph()));
}
