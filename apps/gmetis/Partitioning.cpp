#include "GMetisConfig.h"
#include "MetisGraph.h"
#include "Metis.h"

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

  auto flag = Galois::MethodFlag::NONE;

  do {
    //pick a seed
    int i = getRandom((oldPart.partSize-newSize)/2)+1;
    for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii)
      if (g.getData(*ii, flag).getTryPart(newPart.tryPart) == oldPart.partNum) {
	if(--i) {
          boundary.push_back(*ii);
          break;
	}
      }
    
    //grow partition
    while (newWeight < targetWeight && !boundary.empty()) {
      GNode n =  boundary.front();
      boundary.pop_front();
      if (g.getData(n, flag).getTryPart(newPart.tryPart) == newPart.partNum)
        continue;
      newWeight += g.getData(n, flag).getWeight();
      g.getData(n, flag).setTryPart(newPart.tryPart, newPart.partNum);
      newSize++;
      for (auto ii = g.edge_begin(n, flag), ee = g.edge_end(n, flag); ii != ee; ++ii)
        if (g.getData(g.getEdgeDst(ii, flag), flag).getTryPart(newPart.tryPart) == oldPart.partNum)
          boundary.push_back(g.getEdgeDst(ii, flag));
    }
  } while (newWeight < targetWeight);
  
  oldPart.partWeight -= newWeight;
  oldPart.partSize -= newSize;
  return newPart;
}

partInfo bisect_GGGP(partInfo& oldPart) {
  partInfo p = bisect_GGP(oldPart);
  refine_FL_pair(oldPart, p);
  return p;
}



parallelBisect::parallelBisect(partInfo** parts): parts(parts) { 
} 
  
void parallelBisect::operator()(partInfo &item, Galois::UserContext<partInfo> &workList) { 
  partInfo newPart = bisect_GGP(item); 
  
  if(item.neededParts >1) 
    workList.push(item); 
  else     
    parts[item.tryPart][item.partNum] = item; 
  if(newPart.neededParts >1) 
    workList.push(newPart);  
  else 
    parts[item.tryPart][newPart.partNum] = newPart; 
} 
