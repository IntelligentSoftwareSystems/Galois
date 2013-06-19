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

void refine_BKL(GGraph& g, std::vector<partInfo>& parts) {
  std::set<GNode> boundary;
  
  //find boundary nodes with positive gain
  for (auto nn = g.begin(), en = g.end(); nn != en; ++nn) {
    unsigned gPart = g.getData(*nn).getPart();
    for (auto ii = g.edge_begin(*nn), ee = g.edge_end(*nn); ii != ee; ++ii)
      if (g.getData(g.getEdgeDst(ii)).getPart() != gPart) {
        int ga = gain(g, *nn);
        if (ga > 0)
          boundary.insert(*nn);
        break;
      }
  }

  //refine by swapping with a neighbor high-gain node
  while (!boundary.empty()) {
    GNode n = *boundary.begin();
    boundary.erase(boundary.begin());
    unsigned nPart = g.getData(n).getPart();
    for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii != ee; ++ii) {
      GNode neigh = g.getEdgeDst(ii);
      unsigned neighPart = g.getData(neigh).getPart();
      if (neighPart != nPart && boundary.count(neigh) &&
          gain(g, n) > 0 && gain(g, neigh) > 0 ) {
        unsigned nWeight = g.getData(n).getWeight();
        unsigned neighWeight = g.getData(neigh).getWeight();
        //swap
        g.getData(n).setPart(neighPart);
        g.getData(neigh).setPart(nPart);
        //update partinfo
        parts[neighPart].partWeight += nWeight;
        parts[neighPart].partWeight -= neighWeight;
        parts[nPart].partWeight += neighWeight;
        parts[nPart].partWeight -= nWeight;
        //remove nodes
        boundary.erase(neigh);
        break;
      }
    }
  }
}

struct projectPart {
  GGraph* fineGraph;
  GGraph* coarseGraph;
  projectPart(MetisGraph* Graph) :fineGraph(Graph->getFinerGraph()->getGraph()), coarseGraph(Graph->getGraph()) {}

  void operator()(GNode n) {
    auto& cn = coarseGraph->getData(n);
    unsigned part = cn.getPart();
    for (unsigned x = 0; x < cn.numChildren(); ++x)
      fineGraph->getData(cn.getChild(0)).setPart(part);
  }
};

void coarsen(MetisGraph* coarseGraph, std::vector<partInfo>& parts) {
  unsigned nparts = parts.size();
  do {
    //refine nparts times
    for (int x = 0; x < nparts; ++x) 
      refine_BKL(*coarseGraph->getGraph(),parts);
    //project up
    MetisGraph* fineGraph = coarseGraph->getFinerGraph();
    if (coarseGraph->getFinerGraph())
      Galois::do_all_local(*coarseGraph->getGraph(), projectPart(coarseGraph), "project");
  } while ((coarseGraph = coarseGraph->getFinerGraph()));
}
