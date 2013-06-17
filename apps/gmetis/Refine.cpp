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
  for (GNode n : g) {
    unsigned gPart = g.getData(n).getPart();
    for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii != ee; ++ii)
      if (g.getData(g.getEdgeDst(ii)).getPart() != gPart) {
        int ga = gain(g,n);
        if (ga > 0)
          boundary.insert(n);
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
