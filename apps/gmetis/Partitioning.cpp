#include "MetisGraph.h"
#include "Partitioning.h"

std::ostream& operator<<(std::ostream& os, const partInfo& p) {
  os << "Num " << p.partNum << "\tmask " << p.partMask << "\tweight " << p.partWeight << "\ttotal " << p.totalWeight << "\tgraph " << p.graph <<" size: " <<p.partSize<< " tryPart: " <<p.tryPart;
  return os;
}

partInfo bisect_GGP(partInfo& oldPart) {
  GGraph& g = *oldPart.graph;
  partInfo newPart = oldPart.split();
  std::deque<GNode> boundary;
  unsigned& newWeight = newPart.partWeight = 0;
  unsigned& newSize = newPart.partSize;
  newSize =0;
  unsigned targetWeight = oldPart.partWeight *(newPart.neededParts)/(newPart.neededParts + oldPart.neededParts);
  //pick a seed

  do {

    //pick a seed
    for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii)
      if (g.getData(*ii, Galois::MethodFlag::NONE).getTryPart(newPart.tryPart) == oldPart.partNum) {
        boundary.push_back(*ii);
        break;
      }
    
    //grow partition
    while (newWeight < targetWeight && !boundary.empty()) {
      GNode n =  boundary.front();
      boundary.pop_front();
      if (g.getData(n, Galois::MethodFlag::NONE).getTryPart(newPart.tryPart) == newPart.partNum)
        continue;
      newWeight += g.getData(n, Galois::MethodFlag::NONE).getWeight();
      g.getData(n, Galois::MethodFlag::NONE).setTryPart(newPart.tryPart, newPart.partNum);
      	   newSize ++;
      for (auto ii = g.edge_begin(n, Galois::MethodFlag::NONE), ee = g.edge_end(n, Galois::MethodFlag::NONE); ii != ee; ++ii)
        if (g.getData(g.getEdgeDst(ii, Galois::MethodFlag::NONE), Galois::MethodFlag::NONE).getTryPart(newPart.tryPart) == oldPart.partNum)
          boundary.push_back(g.getEdgeDst(ii, Galois::MethodFlag::NONE));
    }
  } while (newWeight < targetWeight);

  oldPart.partWeight -= newWeight;
  oldPart.partSize -= newSize;
  return newPart;
}



