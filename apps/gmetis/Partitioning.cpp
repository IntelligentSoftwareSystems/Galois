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

#include "Metis.h"

#include <cstdlib>

const bool multiSeed = true;

namespace {

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


GNode findSeed(GGraph& g, unsigned partNum, Galois::MethodFlag flag) {
  //pick a seed
  GNode seed;
  bool validSeed = false;
  for (auto ii = g.begin(), ee = g.end(); ii != ee; ++ii) {
    if (g.getData(*ii, flag).getPart() == partNum) {
      if (!validSeed) {
        validSeed = true;
        seed = *ii;
      } else if (drand48() < 0.01) {
        seed = *ii;
      }
    }
  }
  assert(validSeed);
  return seed;
}

struct bisect_GGP {
  partInfo operator()(GGraph& g, partInfo& oldPart, std::pair<unsigned, unsigned> ratio) {
    partInfo newPart = oldPart.split();
    std::deque<GNode> boundary;
    unsigned& newWeight = newPart.partWeight = 0;
    unsigned targetWeight = oldPart.partWeight * ratio.second / (ratio.first + ratio.second);

    auto flag = Galois::MethodFlag::NONE;

    do {
      boundary.push_back(findSeed(g, oldPart.partNum, flag));

      //grow partition
      while (newWeight < targetWeight && !boundary.empty()) {
        GNode n =  boundary.front();
        boundary.pop_front();
        if (g.getData(n, flag).getPart() == newPart.partNum)
          continue;
        newWeight += g.getData(n, flag).getWeight();
        g.getData(n, flag).setPart(newPart.partNum);
        for (auto ii = g.edge_begin(n, flag), ee = g.edge_end(n, flag); ii != ee; ++ii)
          if (g.getData(g.getEdgeDst(ii, flag), flag).getPart() == oldPart.partNum)
            boundary.push_back(g.getEdgeDst(ii, flag));
      }
    } while (newWeight < targetWeight && multiSeed);
  
    oldPart.partWeight -= newWeight;
    return newPart;
  }
};

struct bisect_GGGP {
  partInfo operator()(GGraph& g, partInfo& oldPart, std::pair<unsigned, unsigned> ratio) {
    partInfo newPart = oldPart.split();
    std::map<GNode, int> gains;
    std::map<int, std::set<GNode>> boundary;

    unsigned& newWeight = newPart.partWeight = 0;
    unsigned targetWeight = oldPart.partWeight * ratio.second / (ratio.first + ratio.second);
    //pick a seed

    auto flag = Galois::MethodFlag::NONE;

    do {
      boundary[0].insert(findSeed(g, oldPart.partNum, flag));
    
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

} //anon namespace
  


  
std::vector<partInfo> partition(MetisGraph* mcg, unsigned numPartitions, InitialPartMode partMode) {
  std::vector<partInfo> parts(numPartitions);
  parts[0] = partInfo(mcg->getTotalWeight());
 switch (partMode) {
    case GGP:
      std::cout <<"Sarting initial partitioning using GGP:\n";
      Galois::for_each<Galois::WorkList::ChunkedLIFO<1> >(&parts[0], parallelBisect<bisect_GGP>(mcg, numPartitions, parts));
      break;
    case GGGP: 
      std::cout <<"Sarting initial partitioning using GGGP:\n";
      Galois::for_each<Galois::WorkList::ChunkedLIFO<1> >(&parts[0], parallelBisect<bisect_GGGP>(mcg, numPartitions, parts));
      break;
    default: abort();
  }

  if (!multiSeed) {
    printPartStats(parts);
    unsigned maxWeight = 1.01 * mcg->getTotalWeight() / numPartitions;
    balance(mcg, parts, maxWeight);
  }

  return parts;
}



int computeCut(GGraph& g) {
  int cuts=0;

 //find boundary nodes with positive gain
  for (auto nn = g.begin(), en = g.end(); nn != en; ++nn) {
    unsigned gPart = g.getData(*nn).getPart();
    for (auto ii = g.edge_begin(*nn), ee = g.edge_end(*nn); ii != ee; ++ii) {
      auto& m = g.getData(g.getEdgeDst(ii));
      if (m.getPart() != gPart) {
        cuts += g.getEdgeData(ii);
      }
    }
  }

  return cuts/2;
}


std::vector<partInfo> BisectAll(MetisGraph* mcg, unsigned numPartitions)
{
  std::cout <<"Sarting initial partitioning using MGGGP:\n";
  auto flag = Galois::MethodFlag::NONE;
  int meanWeight= mcg->getTotalWeight()/numPartitions;
  GGraph& g = *mcg->getGraph();

  int bestCut = mcg->getTotalWeight();
  std::map<GNode, int> bestParts;
  std::vector<partInfo> bestPartInfos(numPartitions);

  for(int nbTry =0; nbTry <20; nbTry ++){
	  std::vector<partInfo> partInfos(numPartitions);

	  std::vector<std::map<GNode, int> > gains(numPartitions);
	  std::vector<std::map<int, std::set<GNode>>> boundary(numPartitions);
	  std::map<int, std::set<int>> partitions;
	  for(GGraph::iterator ii = g.begin(),ee = g.end();ii!=ee;ii++) 
	    g.getData(*ii).setPart(-1);
	  auto seedIter = g.begin();
	  int k =0;
	  for (int i =0; i<numPartitions; i++){  
	    int seed = (int)(drand48()*(mcg->getNumNodes())) +1;
	    bool goodseed = true;
	    while(seed--)
	      if(++seedIter== g.end())seedIter = g.begin();
	    GNode n = *seedIter;

	    for(int j=0; j<i && k <mcg->getNumNodes()/2; j++){
	      goodseed = goodseed && (*boundary[j][0].begin() != n);
	      for (auto ii = g.edge_begin(n, flag), ee = g.edge_end(n, flag); ii != ee; ++ii) 
		goodseed = goodseed && (*boundary[j][0].begin() !=  g.getEdgeDst(ii, flag));
	    }
	    if (!goodseed){
	      k++;
	      i--;
	      continue;
	    }
	    gains[i][n] = 0;
	    partInfos[i] = partInfo(i, 0, 0);
	    boundary[i][0].insert(n);
	    partitions[0].insert(i);
	  }
	  while(!partitions.empty()){
	    //find next partition to improove 
	    auto bb = partitions.begin();
	    int partToMod = *bb->second.begin();
	    bb->second.erase(bb->second.begin());
	    if (bb->second.empty())
	      partitions.erase(bb->first);
	    //node to add to the partition
	    GNode n;
	    if(boundary[partToMod].empty()){
	      GGraph::iterator ii = g.begin(),ee = g.end();
	      for(;ii!=ee;ii++)
		if(g.getData(*ii).getPart() == -1) break; 
	      if (ii == ee) break;
	      else n = *ii;
	    }
	    else{
	      auto bi = boundary[partToMod].rbegin();
	      n =  *bi->second.begin();
	      bi->second.erase(bi->second.begin());
	      if (bi->second.empty())
		boundary[partToMod].erase(bi->first);
	    }
	    if (g.getData(n, flag).getPart() != -1){
	      partitions[partInfos[partToMod].partWeight].insert(partToMod);
	      continue;}
	    //add the node
	     partInfos[partToMod].partWeight += g.getData(n, flag).getWeight();
	     //if(partInfos[partToMod].partWeight < meanWeight) 
	       partitions[partInfos[partToMod].partWeight].insert(partToMod);
	     g.getData(n, flag).setPart(partToMod);
	     for (auto ii = g.edge_begin(n, flag), ee = g.edge_end(n, flag); ii != ee; ++ii) {
		GNode dst = g.getEdgeDst(ii, flag);
		/*auto gi = gains[partToMod].find(dst);
		if (gi != gains[partToMod].end()) { //update
		  boundary[partToMod][gi->second].erase(dst);
		  if (boundary[gi->second].empty())
		    boundary[partToMod].erase(gi->second);
		  gains[partToMod].erase(dst);
		}*/
		int newgain = gains[partToMod][dst] = gain_limited(g, dst, partToMod, flag);
		boundary[partToMod][newgain].insert(dst);
	     }
	  }
    int newCut = computeCut(g);
    if(newCut<bestCut){
      bestCut = newCut;
      for(GGraph::iterator ii = g.begin(),ee = g.end();ii!=ee;ii++) 
        bestParts[*ii] = g.getData(*ii,flag).getPart();
      for (int i =0; i<numPartitions; i++)
        bestPartInfos[i] = partInfos[i];
    }   
  }

  for(GGraph::iterator ii = g.begin(),ee = g.end();ii!=ee;ii++) 
     g.getData(*ii,flag).setPart(bestParts[*ii]);
  
  return bestPartInfos;
}



