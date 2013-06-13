#include "MetisGraph.h"
#include "Partitioning.h"

std::ostream& operator<<(std::ostream& os, const partInfo& p) {
  os << "Num " << p.partNum << "\tmask " << p.partMask << "\tweight " << p.partWeight << "\ttotal " << p.totalWeight << "\tgraph " << p.graph;
  return os;
}


partInfo bisect_GGP(partInfo& oldPart) {
  GGraph& g = *oldPart.graph;
  partInfo newPart = oldPart.split();

  std::deque<GNode> boundary;
  unsigned& newWeight = newPart.partWeight = 0;
  unsigned targetWeight = oldPart.partWeight / 2;

  do {

    //pick a seed
    for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii)
      if (g.getData(*ii).getPart() == oldPart.partNum) {
        boundary.push_back(*ii);
        break;
      }
    
    //grow partition
    while (newWeight < targetWeight && !boundary.empty()) {
      GNode n =  boundary.front();
      boundary.pop_front();
      if (g.getData(n).getPart() == newPart.partNum)
        continue;
      newWeight += g.getData(n).getWeight();
      g.getData(n).setPart(newPart.partNum);
      
      for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii != ee; ++ii)
        if (g.getData(g.getEdgeDst(ii)).getPart() == oldPart.partNum)
          boundary.push_back(g.getEdgeDst(ii));
    }
  } while (newWeight < targetWeight);

  oldPart.partWeight -= newWeight;

  return newPart;
}

