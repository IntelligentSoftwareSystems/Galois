#include "GMetisConfig.h"
#include "MetisGraph.h"
#include "Metis.h"

#include <iomanip>

const bool multiSeed = true;

std::ostream& operator<<(std::ostream& os, const partInfo& p) {
  os << "Num " << std::setw(3) << p.partNum << "\tmask " << std::setw(5) << p.partMask << "\tweight " << p.partWeight << " size: " << p.partSize;
  return os;
}

//gain of moving n from it's current part to new part
int gain_limited(GGraph& g, GNode n, unsigned newpart, Galois::MethodFlag flag) {
  int retval = 0;
  unsigned nPart = g.getData(n,flag).getPart();
  for (auto ii = g.edge_begin(n,flag), ee =g.edge_end(n,flag); ii != ee; ++ii) {
    GNode neigh = g.getEdgeDst(ii,flag);
    if (g.getData(neigh,flag).getPart() == nPart)
      retval -= g.getEdgeData(ii,flag);
    else if (g.getData(neigh,flag).getPart() == newpart)
      retval += g.getEdgeData(ii,flag);
  }
  return retval;
}


struct bisect_GGP {
  partInfo operator()(GGraph& g, partInfo& oldPart, std::pair<unsigned, unsigned> ratio) {
    partInfo newPart = oldPart.split();
    std::deque<GNode> boundary;
    unsigned& newWeight = newPart.partWeight = 0;
    unsigned& newSize = newPart.partSize;
    newSize =0;
    unsigned targetWeight = oldPart.partWeight * ratio.second / (ratio.first + ratio.second);
    //pick a seed

    auto flag = Galois::MethodFlag::NONE;

    do {
      //pick a seed
      int i = getRandom((oldPart.partSize-newSize)/2)+1;
      for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii)
        if (g.getData(*ii, flag).getPart() == oldPart.partNum) {
          if(--i) {
            boundary.push_back(*ii);
            break;
          }
        }
    
      //grow partition
      while (newWeight < targetWeight && !boundary.empty()) {
        GNode n =  boundary.front();
        boundary.pop_front();
        if (g.getData(n, flag).getPart() == newPart.partNum)
          continue;
        newWeight += g.getData(n, flag).getWeight();
        g.getData(n, flag).setPart(newPart.partNum);
        newSize++;
        for (auto ii = g.edge_begin(n, flag), ee = g.edge_end(n, flag); ii != ee; ++ii)
          if (g.getData(g.getEdgeDst(ii, flag), flag).getPart() == oldPart.partNum)
            boundary.push_back(g.getEdgeDst(ii, flag));
      }
    } while (newWeight < targetWeight && multiSeed);
  
    oldPart.partWeight -= newWeight;
    oldPart.partSize -= newSize;
    return newPart;
  }
};

struct bisect_GGGP {
  partInfo operator()(GGraph& g, partInfo& oldPart, std::pair<unsigned, unsigned> ratio) {
    partInfo newPart = oldPart.split();
    std::map<GNode, int> gains;
    std::map<int, std::set<GNode>> boundary;

    unsigned& newWeight = newPart.partWeight = 0;
    unsigned& newSize = newPart.partSize = 0;
    unsigned targetWeight = oldPart.partWeight * ratio.second / (ratio.first + ratio.second);
    //pick a seed

    auto flag = Galois::MethodFlag::NONE;

    do {
      //pick a seed
      int i = getRandom((oldPart.partSize-newSize)/2)+1;
      for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii)
        if (g.getData(*ii, flag).getPart() == oldPart.partNum) {
          if(--i) {
            boundary[0].insert(*ii);
            break;
          }
        }
    
      //grow partition
      while (newWeight < targetWeight && !boundary.empty()) {
        auto bi = boundary.rbegin();
        GNode n =  *bi->second.begin();
        bi->second.erase(bi->second.begin());
        if (bi->second.empty())
          boundary.erase(bi->first);
        if (g.getData(n, flag).getPart() == newPart.partNum)
          continue;
        newWeight += g.getData(n, flag).getWeight();
        newSize++;
        g.getData(n, flag).setPart(newPart.partNum);
        for (auto ii = g.edge_begin(n, flag), ee = g.edge_end(n, flag); ii != ee; ++ii) {
          GNode dst = g.getEdgeDst(ii, flag);
          auto gi = gains.find(dst);
          if (gi != gains.end()) { //update
            boundary[gi->second].erase(dst);
            if (boundary[gi->second].empty())
              boundary.erase(gi->second);
            gains.erase(dst);
          }
          if (g.getData(dst, flag).getPart() == oldPart.partNum) {
            int newgain = gains[dst] = gain_limited(g, dst, newPart.partNum, flag);
            boundary[newgain].insert(dst);
          }
        }
      }
    } while (newWeight < targetWeight && multiSeed);
  
    oldPart.partWeight -= newWeight;
    oldPart.partSize -= newSize;
    return newPart;
  }
};


template<typename bisector>
struct parallelBisect { 
  unsigned totalWeight;
  unsigned nparts;
  GGraph* graph;
  bisector bisect;
  std::vector<partInfo>& parts;

  parallelBisect(MetisGraph* mg, unsigned parts, std::vector<partInfo>& pb, bisector b = bisector())
    :totalWeight(mg->getTotalWeight()), nparts(parts), graph(mg->getGraph()), bisect(b), parts(pb)
  {}
  void operator()(partInfo* item, Galois::UserContext<partInfo*> &cnx) {
    if (item->splitID() >= nparts) //when to stop
      return;
    std::pair<unsigned, unsigned> ratio = item->splitRatio(nparts);
    //std::cout << "Splitting " << item->partNum << ":" << item->partMask << " L " << ratio.first << " R " << ratio.second << "\n";
    partInfo newPart = bisect(*graph, *item, ratio);
    //std::cout << "Result " << item->partNum << " " << newPart.partNum << "\n";
    parts[newPart.partNum] = newPart;
    cnx.push(&parts[newPart.partNum]);
    cnx.push(item);
  }
}; 
  
std::vector<partInfo> partition(MetisGraph* mcg, unsigned numPartitions) {
  std::vector<partInfo> parts(numPartitions);
  parts[0] = partInfo(mcg->getTotalWeight(), mcg->getNumNodes());
  Galois::for_each<Galois::WorkList::GFIFO<> >(&parts[0], parallelBisect<bisect_GGGP>(mcg, numPartitions, parts));
  return parts;
}
