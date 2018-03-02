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

#include "galois/Galois.h"
#include "galois/Timer.h"
#include "Metis.h"
#include <set>
#include "galois/Galois.h"
#include <map>
#include <set>
#include <cstdlib>
#include <iostream>
#include <stack>
#include <climits>
const bool multiSeed = true;

namespace {

//gain of moving n from it's current part to new part
int gain_limited(GGraph& g, GNode n, unsigned newpart, galois::MethodFlag flag) {
  int retval = 0;
  unsigned nPart = g.getData(n,flag).getPart();
  for (auto ii : g.edges(n, flag)) {
    GNode neigh = g.getEdgeDst(ii);
    auto nData = g.getData(neigh,flag);
    if (nData.getPart() == nPart)
      retval -= g.getEdgeData(ii,flag);
    else if (nData.getPart() == newpart)
      retval += g.getEdgeData(ii,flag);
  }
  return retval;
}


GNode findSeed(GGraph& g, unsigned partNum, int partWeight, galois::MethodFlag flag) {
  //pick a seed
  
  int rWeight = (int)(drand48()*(partWeight));
  GNode seed;
  /*std::vector<std::pair<int,GNode> > nodeEd;
  for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii) {
    if (g.getData(*ii, flag).getPart() == partNum) {
      seed = *ii;
      nodeEd.push_back(std::make_pair(std::distance(g.edge_begin(*ii),g.edge_end(*ii)),*ii));
    }
  }
  std::sort(nodeEd.begin(),nodeEd.end());
  return nodeEd[nodeEd.size()-1].second;
  */
  for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii) {
    if (g.getData(*ii, flag).getPart() == partNum) {
      seed = *ii;
      rWeight -= g.getData(*ii, flag).getWeight();
      if(rWeight <0) return *ii;
    }
  }

   return seed;
}


struct bisect_GGP {
  partInfo operator()(GGraph& g, partInfo& oldPart, std::pair<unsigned, unsigned> ratio,std::vector<GNode> *b = NULL, int oldWeight = 0) {
    partInfo newPart = oldPart.split();
    std::deque<GNode> boundary;
    unsigned& newWeight = newPart.partWeight = 0;
    unsigned targetWeight = oldPart.partWeight * ratio.second / (ratio.first + ratio.second);

    auto flag = galois::MethodFlag::UNPROTECTED;

    do {
      boundary.push_back(findSeed(g, oldPart.partNum, oldPart.partWeight, flag));
      //grow partition
      while (newWeight < targetWeight && !boundary.empty()) {
        GNode n =  boundary.front();
        boundary.pop_front();
        if (g.getData(n, flag).getPart() == newPart.partNum)
          continue;
        newWeight += g.getData(n, flag).getWeight();
        g.getData(n, flag).setPart(newPart.partNum);
        if(b) b->push_back(n);
        for (auto ii : g.edges(n, flag))
          if (g.getData(g.getEdgeDst(ii), flag).getPart() == oldPart.partNum)
            boundary.push_back(g.getEdgeDst(ii));
      }
    } while (newWeight < targetWeight && multiSeed);

    oldPart.partWeight -= newWeight;
    return newPart;
  }
};


struct bisect_GGGP {
  partInfo operator()(GGraph& g, partInfo& oldPart, std::pair<unsigned, unsigned> ratio,std::vector<GNode> *b = NULL, int oldWeight = 0) {
    partInfo newPart = oldPart.split();
    std::map<GNode, int> gains;
    std::map<int, std::set<GNode>> boundary;

    unsigned& newWeight = newPart.partWeight = oldWeight;
    unsigned targetWeight = oldPart.partWeight * ratio.second / (ratio.first + ratio.second);
    //pick a seed

    auto flag = galois::MethodFlag::UNPROTECTED;

    do {
      //boundary[0].insert(findSeed(g, oldPart.partNum, oldPart.partWeight, flag));
      GNode bNode = findSeed(g, oldPart.partNum, oldPart.partWeight, flag);
      boundary[gain_limited(g,bNode, newPart.partNum, flag)].insert(bNode);

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
        g.getData(n, flag).setPart(newPart.partNum);
        if(b) b->push_back(n);
        for (auto ii : g.edges(n, flag)) {
          GNode dst = g.getEdgeDst(ii);
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
    return newPart;
  }
};

int computeEdgeCut(GGraph& g) {
  int cuts=0;
  for (auto nn : g) {
    unsigned gPart = g.getData(nn).getPart();
    for (auto ii :g.edges(nn)) {
      auto& m = g.getData(g.getEdgeDst(ii));
      if (m.getPart() != gPart) {
        cuts += g.getEdgeData(ii);
      }
    }
  }

  return cuts/2;
}
int node_gain(GGraph &graph, GNode node) {
  auto nData = graph.getData(node,galois::MethodFlag::UNPROTECTED); 
  int gain = 0;
  for (auto ei : graph.edges(node)) {
    auto neigh = graph.getEdgeDst(ei); 
    int ew = graph.getEdgeData(ei); 
    auto neighData = graph.getData(neigh,galois::MethodFlag::UNPROTECTED);
    if (nData.getPart() != neighData.getPart()) { 
      gain += ew;
    } else {
      gain -= ew;
    }
  }
  return gain;
}

typedef std::pair<int,std::pair<GNode,GNode> > PartMatch;
typedef galois::substrate::PerThreadStorage<PartMatch > PerThreadPartInfo;
struct KLMatch {
  GGraph &graph;
  PerThreadPartInfo &threadInfo;
  int oldPartNum;
  int newPartNum;
  KLMatch(GGraph &g,PerThreadPartInfo &ti, 
    int oldPartNum, int newPartNum):graph(g),threadInfo(ti) {
    this->oldPartNum = oldPartNum;
    this->newPartNum = newPartNum;
  }
  bool isPartOk(int partNum) { 
    return (partNum == oldPartNum || partNum == newPartNum);
  }
  bool isNodeOk(MetisNode &node) { 
    return !node.isLocked() && isPartOk(node.getPart());
  }
  template<typename Context>
  void operator()(GNode node, Context& lwl) {
    auto flag = galois::MethodFlag::UNPROTECTED;
    PartMatch *localInfo = threadInfo.getLocal();
    int gain = localInfo->first;
    auto& srcData = graph.getData(node,flag);
    int srcGain = 0;
    if (!isNodeOk(srcData)) {
      return;
    }
    
    for (auto ei : graph.edges(node, flag)) {
      int ew = graph.getEdgeData(ei,flag);
      GNode n = graph.getEdgeDst(ei);
      auto& nData = graph.getData(n,flag);
      if (!isNodeOk(nData)) {
        continue;
      }
      if (nData.getPart() == srcData.getPart()) {
        srcGain -= ew;
      } else {
        srcGain += ew;
      }
    }
    for (auto ei : graph.edges(node, flag)) {
      GNode n = graph.getEdgeDst(ei);
      auto nData = graph.getData(n,flag);
      int nw = graph.getEdgeData(ei,flag);
      int neighGain = 0;
      if (!isNodeOk(nData) || nData.getPart() == srcData.getPart()) {
        continue;
      }
      for (auto nei : graph.edges(n, flag)) {
        int ew = graph.getEdgeData(nei,flag);
        GNode nn = graph.getEdgeDst(nei);
        auto nnData = graph.getData(nn,flag);
        if (!isNodeOk(nnData)) {
          continue;
        }
        if (nnData.getPart() == nData.getPart()) {
          neighGain -= ew;
        } else {
          neighGain += ew;
        }
      }
      int totalGain = srcGain + neighGain - 2 * nw;
      if ( totalGain > gain) { 
        gain = totalGain;
        localInfo->first = gain;
        localInfo->second.first = node;
        localInfo->second.second = n;
      }
    }
    
  }
};
void refine_kl(GGraph &graph,std::vector<GNode> &boundary,int oldPartNum, int newPartNum,std::vector<partInfo>& parts) {
  std::vector<GNode> swappedNodes;
  std::vector<PartMatch> foundNodes;
  //int iter = 0;
  do {
    std::vector<PartMatch> matches;
    for (unsigned int j = 0; j < boundary.size(); j++) {
      PerThreadPartInfo iterationInfo;
      for (unsigned int i = 0; i < iterationInfo.size(); i++) {
        iterationInfo.getRemote(i)->first = INT_MIN;
        iterationInfo.getRemote(i)->second.first = NULL; 
        iterationInfo.getRemote(i)->second.second = NULL; 
      }
      KLMatch matchIter(graph,iterationInfo,oldPartNum,newPartNum);
      galois::for_each(galois::iterate(boundary), 
          matchIter,
          galois::loopname("KLMatch"),
          galois::wl<galois::worklists::ChunkedLIFO<32> >());
      PartMatch bestMatch;
      bestMatch.first = INT_MIN;
      for (unsigned int i = 0; i < iterationInfo.size(); i++) {
        PartMatch match = *iterationInfo.getRemote(i);
        if (match.first > bestMatch.first) {
          bestMatch = match;
        }
      }
      if (bestMatch.second.first == NULL || bestMatch.second.second == NULL) 
        break;
      auto& m1 = graph.getData(bestMatch.second.first);
      auto& m2 = graph.getData(bestMatch.second.second);
      m1.setLocked(true);
      m2.setLocked(true);
      matches.push_back(bestMatch);
      foundNodes.push_back(bestMatch);
    }
    if (matches.size() == 0) {
      break;
    }
    int g_max = 0;
    int temp = 0;
    int index = -1;
    for (unsigned int k = 0; k < matches.size(); k++) { 
      g_max+= matches[k].first;
      if ( g_max > temp) { 
        temp = g_max; 
        index = k;
      }
    }
    g_max = temp;

    if (g_max <=0 || index < 0) 
      break;
    
    for (int i= 0; i <= index; i++) { 
      PartMatch match = matches[i];
      GNode n1 = match.second.first; 
      GNode n2 = match.second.second;
      auto& n1Data = graph.getData(n1); 
      auto& n2Data = graph.getData(n2);
      int p1 = n1Data.getPart(); 
      int p2 = n2Data.getPart();
      parts[p1].partWeight += (n2Data.getWeight() - n1Data.getWeight());
      parts[p2].partWeight += (n1Data.getWeight() - n2Data.getWeight());
      n1Data.setPart(p2);
      n2Data.setPart(p1);
      swappedNodes.push_back(n1);
      swappedNodes.push_back(n2);
    }
    for (unsigned int i = index+1; i<matches.size(); i++) { 
      auto& m1 = graph.getData(matches[i].second.first);
      auto& m2 = graph.getData(matches[i].second.second);
      m1.setLocked(false);
      m2.setLocked(false);
    }
  }while(true);
  for (PartMatch match : foundNodes) {
    graph.getData(match.second.first).setLocked(false);
    graph.getData(match.second.second).setLocked(false);
  }
}

template<typename bisector>
struct serialBisect {
  unsigned totalWeight;
  unsigned nparts;
  GGraph* graph;
  bisector bisect;
  std::vector<partInfo>& parts;
  std::stack<partInfo*> workList;
  MetisGraph *mcg;
  serialBisect(MetisGraph* mg, unsigned parts, std::vector<partInfo>& pb, bisector b = bisector())
    :totalWeight(mg->getTotalWeight()), nparts(parts), graph(mg->getGraph()), bisect(b), parts(pb)
  {
    workList.push(&(this->parts[0]));
    mcg = mg;
  }
  void operator()() {
    while(!workList.empty())  {
     partInfo *item = workList.top();
     workList.pop();
     if (item->splitID() >= nparts) //when to stop
      continue;
     std::pair<unsigned, unsigned> ratio = item->splitRatio(nparts);
     std::vector<GNode> newNodes;
     //int iter = 0;
     partInfo newPart;
     newPart.partWeight = 0;
     newPart = bisect(*graph, *item, ratio,&newNodes,newPart.partWeight);
     parts[newPart.partNum] = newPart;
     refine_kl(*graph,newNodes,item->partNum,newPart.partNum,parts);
     newPart.partWeight = parts[newPart.partNum].partWeight; 
     item->partWeight = parts[item->partNum].partWeight;
     //unsigned targetWeight = item->partWeight * ratio.second / (ratio.first + ratio.second);
     item->partWeight = parts[item->partNum].partWeight; 
     workList.push(&(parts[newPart.partNum]));
     workList.push(item);
    }
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
  void operator()(partInfo* item, galois::UserContext<partInfo*> &cnx) {
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

} //anon namespace


std::vector<partInfo> partition(MetisGraph* mcg, unsigned numPartitions, InitialPartMode partMode) {
  std::vector<partInfo> parts(numPartitions);
  parts[0] = partInfo(mcg->getTotalWeight());

  galois::do_all(galois::iterate(*mcg->getGraph()), 
      [g=mcg->getGraph()] (GNode item) {
        g->getData(item, galois::MethodFlag::UNPROTECTED).initRefine(0,true);
        g->getData(item, galois::MethodFlag::UNPROTECTED).initPartition();
      },
      galois::loopname("initPart"));

  bool serialPartition = false;
  if (serialPartition) {
  serialBisect<bisect_GGGP>(mcg, numPartitions, parts)();
  } else {
    switch (partMode) {
      case GGP:
        std::cout <<"\nSorting initial partitioning using GGP:\n";
        galois::for_each(galois::iterate( {&parts[0]} ), 
            parallelBisect<bisect_GGP>(mcg, numPartitions, parts), 
            galois::loopname("parallelBisect"),
            galois::wl<galois::worklists::ChunkedLIFO<1>>());
        break;
      case GGGP:
        std::cout <<"\nSorting initial partitioning using GGGP:\n";
        galois::for_each(galois::iterate( {&parts[0]} ), 
            parallelBisect<bisect_GGGP>(mcg, numPartitions, parts), 
            galois::loopname("parallelBisect"),
            galois::wl<galois::worklists::ChunkedLIFO<1>>());
        break;
      default: abort();
    }
  }
  // XXX(ddn): Leave commented out until we have balance() defined.
#if 0
  if (!multiSeed) {
    unsigned maxWeight = 1.01 * mcg->getTotalWeight() / numPartitions;
    balance(mcg, parts, maxWeight);
  }
#endif
  static_assert(multiSeed, "not yet implemented");
  return parts;
}
namespace {
int edgeCount(GGraph& g) {
  int count=0;
  for (auto nn : g)
    for (auto ii : g.edges(nn)) 
      count += g.getEdgeData(ii);
  return count/2;
}
}
std::vector<partInfo> BisectAll(MetisGraph* mcg, unsigned numPartitions, unsigned maxSize)
{
  std::cout <<"\nSorting initial partitioning using MGGGP:\n";
  auto flag = galois::MethodFlag::UNPROTECTED;
  GGraph& g = *mcg->getGraph();

  int bestCut = edgeCount(g);
  std::map<GNode, int> bestParts;
  std::vector<partInfo> bestPartInfos(numPartitions);

  for(int nbTry =0; nbTry <20; nbTry ++){
    std::vector<partInfo> partInfos(numPartitions);
    std::vector<std::map<int, std::set<GNode>>> boundary(numPartitions);
    std::map<int, std::set<int>> partitions;
    for(auto ii : g) 
      g.getData(ii).setPart(numPartitions+1);
    auto seedIter = g.begin();
    int k =0;
    //find one seed for each partition and do initialization 
    for (unsigned int i =0; i<numPartitions; i++){
      int seed = (int)(drand48()*(mcg->getNumNodes())) +1;
      bool goodseed = true;
      while(seed--)
        if(++seedIter== g.end())seedIter = g.begin();
      GNode n = *seedIter;

      for(unsigned int j=0; j<i && k <50; j++){
        goodseed = goodseed && (*boundary[j][0].begin() != n);
        for (auto ii : g.edges(n, flag))
          goodseed = goodseed && (*boundary[j][0].begin() !=  g.getEdgeDst(ii));
      }
      if (!goodseed){
        k++;
        i--;
        continue;
      }
      partInfos[i] = partInfo(i, 0, 0);
      boundary[i][0].insert(n);
      partitions[0].insert(i);
    }
          auto beg = g.begin();
    while(!partitions.empty()){
      //find the next partition to improove
      auto bb = partitions.begin();
      int partToMod = *bb->second.begin();
      bb->second.erase(bb->second.begin());
      if (bb->second.empty())
        partitions.erase(bb->first);

      //find the node to add to the partition
      GNode n = *g.begin();
      do{
        if(boundary[partToMod].empty()) break;
        auto bi = boundary[partToMod].rbegin();
        n =  *bi->second.begin();
        bi->second.erase(bi->second.begin());
        if (bi->second.empty())
          boundary[partToMod].erase(bi->first);
      }while(g.getData(n, flag).getPart() <numPartitions && !boundary[partToMod].empty());

      if(g.getData(n, flag).getPart() <numPartitions && boundary[partToMod].empty()){
        GGraph::iterator ii = beg, ee = g.end();
        for(;ii!=ee;ii++)
          if(g.getData(*ii).getPart() == numPartitions+1) break;
        if (ii == ee) break;
        else n = *(beg=ii);
      }

      //add the node
      partInfos[partToMod].partWeight += g.getData(n, flag).getWeight();
      partitions[partInfos[partToMod].partWeight].insert(partToMod);
      g.getData(n, flag).setPart(partToMod);
      for (auto ii : g.edges(n, flag)) {
        GNode dst = g.getEdgeDst(ii);
        int newgain= gain_limited(g, dst, partToMod, flag);
        boundary[partToMod][newgain].insert(dst);
      }
    }
    //decides if this partition is the nez best one 
    int newCut = computeEdgeCut(g);
    if(newCut<bestCut){
      bestCut = newCut;
      for(GGraph::iterator ii = g.begin(),ee = g.end();ii!=ee;ii++)
        bestParts[*ii] = g.getData(*ii,flag).getPart();
      for (unsigned int i =0; i<numPartitions; i++)
        bestPartInfos[i] = partInfos[i];
    }
  }

  for(GGraph::iterator ii = g.begin(),ee = g.end();ii!=ee;ii++)
     g.getData(*ii,flag).setPart(bestParts[*ii]);

  return bestPartInfos;
}
