#include "MetisGraph.h"
#include "Partitioning.h"

std::ostream& operator<<(std::ostream& os, const partInfo& p) {
  os << "Num " << p.partNum << " mask " << p.partMask << " weight " << p.partWeight << " total " << p.totalWeight << " graph " << p.graph;
  return os;
}


partInfo bisect_GGP(partInfo& oldPart) {
  GGraph& g = *oldPart.graph;
  partInfo newPart = oldPart.split();

  std::deque<GNode> boundary;
  std::set<GNode> part;
  unsigned& newWeight = newPart.partWeight;
  unsigned targetWeight = oldPart.partWeight / 2;

  //pick a seed
  for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii)
    if (g.getData(*ii).getPart() == oldPart.partNum) {
      boundary.push_back(*ii);
      part.insert(*ii);
      newWeight = g.getData(*ii).getWeight();
      break;
    }

  //grow partition
  while (newWeight < targetWeight && !boundary.empty()) {
    GNode n =  boundary.front();
    boundary.pop_front();
    part.insert(n);
    newWeight += g.getData(n).getWeight();
    g.getData(n).setPart(newPart.partNum);

    for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii != ee; ++ii)
      if (g.getData(g.getEdgeDst(ii)).getPart() == oldPart.partNum)
        boundary.push_back(g.getEdgeDst(ii));
  }

  oldPart.partWeight -= newWeight;

  return newPart;
}

