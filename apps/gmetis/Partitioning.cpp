#include "GMetisConfig.h"
#include "MetisGraph.h"
#include "Metis.h"

#include <iomanip>


std::ostream& operator<<(std::ostream& os, const partInfo& p) {
  os << "Num " << std::setw(3) << p.partNum << "\tmask " << std::setw(5) << p.partMask << "\tweight " << p.partWeight << " size: " << p.partSize;
  return os;
}

//gain of moving n from it's current part to new part
int gain_limited(GGraph& g, GNode n, unsigned newpart) {
  int retval = 0;
  unsigned nPart = g.getData(n).getPart();
  for (auto ii = g.edge_begin(n), ee =g.edge_end(n); ii != ee; ++ii) {
    GNode neigh = g.getEdgeDst(ii);
    if (g.getData(neigh).getPart() == nPart)
      retval -= g.getEdgeData(ii);
    else if (g.getData(neigh).getPart() == newpart)
      retval += g.getEdgeData(ii);
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
    } while (newWeight < targetWeight);
  
    oldPart.partWeight -= newWeight;
    oldPart.partSize -= newSize;
    return newPart;
  }
};


struct gainSorter {
  std::map<GNode, int>& gains;
  gainSorter(std::map<GNode, int>& _g) :gains(_g) {}
  bool operator()(GNode n1, GNode n2) {
    assert(gains.count(n1) && gains.count(n2));
    return gains[n1] > gains[n2];
  }
};

struct bisect_GGGP {
  partInfo operator()(GGraph& g, partInfo& oldPart, std::pair<unsigned, unsigned> ratio) {
    partInfo newPart = oldPart.split();
    std::deque<GNode> boundary;
    std::map<GNode, int> gains;

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
        newSize++;
        g.getData(n, flag).setPart(newPart.partNum);
        for (auto ii = g.edge_begin(n, flag), ee = g.edge_end(n, flag); ii != ee; ++ii) {
          GNode dst = g.getEdgeDst(ii, flag);
          gains[dst] = gain_limited(g, dst, newPart.partNum);
          if (g.getData(dst, flag).getPart() == oldPart.partNum) {
            boundary.push_back(dst);
          }
        }
        std::sort(boundary.begin(), boundary.end(), gainSorter(gains));
      }
    } while (newWeight < targetWeight);
  
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
  Galois::InsertBag<partInfo>& parts;

  parallelBisect(MetisGraph* mg, unsigned parts, Galois::InsertBag<partInfo>& pb, bisector b = bisector())
    :totalWeight(mg->getTotalWeight()), nparts(parts), graph(mg->getGraph()), bisect(b), parts(pb)
  {}
  void operator()(partInfo* item, Galois::UserContext<partInfo*> &cnx) {
    if (item->splitID() >= nparts) //when to stop
      return;
    std::pair<unsigned, unsigned> ratio = item->splitRatio(nparts);
    std::cout << "Splitting " << item->partNum << ":" << item->partMask << " L " << ratio.first << " R " << ratio.second << "\n";
    partInfo newPart = bisect(*graph, *item, ratio);
    cnx.push(&parts.push(newPart));
    cnx.push(item);
  }
}; 
  
std::vector<partInfo> partition(MetisGraph* mcg, unsigned numPartitions) {
  Galois::InsertBag<partInfo> parts;
  partInfo initPart(mcg->getTotalWeight(), mcg->getNumNodes());

  partInfo* p = &parts.push(initPart);
  Galois::for_each(p, parallelBisect<bisect_GGP>(mcg, numPartitions, parts));

  std::vector<partInfo> np(numPartitions);
  for (partInfo& tp : parts)
    np.at(tp.partNum) = tp;
  return np;
}
