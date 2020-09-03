/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "galois/Galois.h"
#include "galois/Timer.h"
#include "Metis.h"
#include <set>
#include "galois/Galois.h"
#include "galois/AtomicHelpers.h"
#include <map>
#include <set>
#include <cstdlib>
#include <iostream>
#include <stack>
#include <climits>
#include <array>
const bool multiSeed = true;

namespace {


// gain of moving n from it's current part to new part
int gain_limited(GGraph& g, GNode n, unsigned newpart,
               galois::MethodFlag flag) {
  int retval     = 0;
  unsigned nPart = g.getData(n).getPart();
  for (auto ii : g.edges(n, flag)) {
    GNode neigh = g.getEdgeDst(ii);
    for (auto i : g.edges(neigh)) {
      auto ineigh = g.getEdgeDst(i);
      auto& nData  = g.getData(ineigh);
      if (nData.getPart() == nPart)
        retval -= nData.getWeight();
      else if (nData.getPart() == newpart)
        retval += nData.getWeight();
      }
    }
  return retval;
}

GNode findSeed(GGraph& g, unsigned partNum, int partWeight,
               galois::MethodFlag flag) {
  // pick a seed

  //int rWeight = (int)(drand48() * (partWeight));
  GNode seed  = *g.cellList().begin();
  for (auto ii = g.cellList().begin(), ee = g.cellList().end(); ii != ee; ++ii) {
    if (g.getData(*ii, flag).getPart() == 0) {
      //seed = *ii;
    //  rWeight -= g.getData(*ii, flag).getWeight();
      //if (rWeight < 0)
        return *ii;
    }
  }

  return seed;
}

using BisectPolicy = partInfo(GGraph& g, partInfo& oldPart,
                              std::pair<unsigned, unsigned> ratio,
                              std::vector<GNode>* b, int oldWeight);

partInfo bisect_GGP(GGraph& g, partInfo& oldPart,
                    std::pair<unsigned, unsigned> ratio,
                    std::vector<GNode>* b = NULL, int oldWeight = 0) {
  partInfo newPart = oldPart.split();
  std::deque<GNode> boundary;
  unsigned& newWeight = newPart.partWeight = 0;
  unsigned targetWeight =
      oldPart.partWeight * ratio.second / (ratio.first + ratio.second);

  auto flag = galois::MethodFlag::UNPROTECTED;

  do {
    boundary.push_back(findSeed(g, oldPart.partNum, oldPart.partWeight, flag));
    // grow partition
    while (newWeight < targetWeight && !boundary.empty()) {
      GNode n = boundary.front();
      boundary.pop_front();
      if (g.getData(n, flag).getPart() == newPart.partNum)
        continue;
      newWeight += g.getData(n, flag).getWeight();
      g.getData(n, flag).setPart(newPart.partNum);
      for (auto ii : g.edges(n, flag)) {
        auto pr = g.getEdgeDst(ii);
        for (auto pp : g.edges(pr, flag)) {
          if (g.getData(g.getEdgeDst(pp), flag).getPart() == oldPart.partNum)
            boundary.push_back(g.getEdgeDst(pp));
        }
      }
    }
  } while (newWeight < targetWeight);

  oldPart.partWeight -= newWeight;
  return newPart;
}


int computeEdgeCut(GGraph& g) {
  int cuts = 0;
  for (auto nn : g) {
    unsigned gPart = g.getData(nn).getPart();
    for (auto ii : g.edges(nn)) {
      auto& m = g.getData(g.getEdgeDst(ii));
      if (m.getPart() != gPart) {
        cuts += g.getEdgeData(ii);
      }
    }
  }

  return cuts / 2;
}

/*int node_gain(GGraph &graph, GNode node) {
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
}*/

int cutsize(GGraph& g) { 
  unsigned size = std::distance(g.cellList().begin(), g.cellList().end());
  unsigned sizen = std::distance(g.getNets().begin(), g.getNets().end());
  int cutsize = 0;
  std::vector<int> cells;
  for (auto n : g.getNets()) { 
    bool cut_status = false;
    for (auto e : g.edges(n)) {
      auto cell1 = g.getEdgeDst(e);
    for (auto c : g.edges(n)) {
        auto cell2 = g.getEdgeDst(c);
        if(g.getData(cell1).getPart() != g.getData(cell2).getPart() && cell1 != cell2) {
          cutsize++;
          cut_status = true;
          break;
        }
      }
      if (cut_status == true)
        break;
    }
  }
  return cutsize;
}

GNode fseed(GGraph& g, unsigned oldpart, int weight, int maxW) {
  int size = std::distance(g.cellList().begin(), g.cellList().end());
  int iter = 0;
  int s;
  std::vector<GNode> node;
  while (iter < size) {
    node.clear();
    iter++;
    for (auto ii : g.cellList()) {
      if (g.getData(ii).getPart() == oldpart && !g.getData(ii).isLocked()) 
        node.push_back(ii);
    }
    s = (int)(drand48() * node.size());
    int w = g.getData(node[s]).getWeight();
    if (w + weight < maxW) break;
  } 
  return node[s];
}

std::pair<GNode, GNode> Jseed(GGraph& g) 
{
  int size = std::distance(g.cellList().begin(), g.cellList().end());
  std::vector<GNode> nodes;
  int tmp[2] = {0,0};
  GNode node_tmp[2];
  node_tmp[0] = *g.cellList().begin();
  node_tmp[1] = *g.cellList().begin() + 1;
  for (auto ii : g.cellList()) {
    if (g.getData(ii).getWeight() > tmp[0]) {
        galois::gstl::Vector<GNode> nets = g.getNets(ii);
      for (auto c : nets) {
        if (g.findEdge(c, ii) != g.edge_end(c)) break;
      }
      tmp[1] = tmp[0];
      node_tmp[1] = node_tmp[0];
      tmp[0] = g.getData(ii).getWeight();
      node_tmp[0] = ii;
    }

  }
  return std::make_pair(node_tmp[0], node_tmp[1]);
}

using BisectPolicy = partInfo(GGraph& g, partInfo& oldPart,
                              std::pair<unsigned, unsigned> ratio,
                              std::vector<GNode>* b, int oldWeight);
partInfo bisect_GGGP(GGraph& g, partInfo& oldPart,
                     std::pair<unsigned, unsigned> ratio,
                     std::vector<GNode>* b = NULL, int oldWeight = 0) {
  partInfo newPart = oldPart.split();
  std::map<GNode, int> gains;
  std::map<int, std::set<GNode>> boundary;

  unsigned& newWeight = newPart.partWeight = oldWeight;
  unsigned targetWeight =
      oldPart.partWeight * ratio.second / (ratio.first + ratio.second);
  // pick a seed

  auto flag = galois::MethodFlag::UNPROTECTED;

  do {
    // boundary[0].insert(findSeed(g, oldPart.partNum, oldPart.partWeight,
    // flag));
    GNode bNode = findSeed(g, oldPart.partNum, oldPart.partWeight, flag);
    boundary[gain_limited(g, bNode, newPart.partNum, flag)].insert(bNode);

    // grow partition
    while (newWeight < targetWeight && !boundary.empty()) {
      auto bi = boundary.rbegin();
      GNode n = *bi->second.begin();
      bi->second.erase(bi->second.begin());
      if (bi->second.empty())
        boundary.erase(bi->first);
      if (g.getData(n, flag).getPart() == newPart.partNum)
        continue;
      newWeight += g.getData(n, flag).getWeight();
      g.getData(n, flag).setPart(newPart.partNum);
      for (auto ii : g.edges(n, flag)) {
        GNode dst = g.getEdgeDst(ii);
        for (auto i : g.edges(dst, flag)) {
          auto idst = g.getEdgeDst(i);
          auto gi   = gains.find(idst);
          if (gi != gains.end()) { // update
            boundary[gi->second].erase(idst);
            if (boundary[gi->second].empty())
              boundary.erase(gi->second);
            gains.erase(idst);
          }
          if (g.getData(dst, flag).getPart() == oldPart.partNum) {
            int newgain = gains[dst] =
                gain_limited(g, dst, newPart.partNum, flag);
            boundary[newgain].insert(dst);
          }
        }
      }
    }
  } while (newWeight < targetWeight && multiSeed);

  oldPart.partWeight -= newWeight;
  return newPart;
}

template <BisectPolicy bisect>
void parallelBisect(MetisGraph* mg, unsigned totalWeight, unsigned nparts,
                    std::vector<partInfo>& parts) {
  GGraph* graph = mg->getGraph();
  galois::for_each(galois::iterate({&parts[0]}),
                   [&](partInfo* item, auto& cnx) {
                     if (item->splitID() >= nparts) // when to stop
                       return;
                     std::pair<unsigned, unsigned> ratio =
                         item->splitRatio(nparts);
                     partInfo newPart = bisect(*graph, *item, ratio, NULL, 0);
                     parts[newPart.partNum] = newPart;
                     cnx.push(&parts[newPart.partNum]);
                     cnx.push(item);
                   },
                   galois::loopname("parallelBisect"),
                   galois::wl<galois::worklists::ChunkLIFO<1>>());
}

void initGain(GGraph& g) {
  galois::do_all(galois::iterate(g.cellList()),
        [&](GNode n) {
            g.getData(n).FS.store(0);
            g.getData(n).TE.store(0);
        },
        galois::loopname("firstinit"));

  galois::do_all(galois::iterate(g.getNets()),
        [&](GNode n) {
            g.getData(n).p1 = 0;
            g.getData(n).p2 = 0;
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              int part = g.getData(cc).getPart();
              if (part == 0) g.getData(n).p1++;
              else g.getData(n).p2++;
            }
        },
        galois::loopname("firstinitGain"));

  galois::do_all(galois::iterate(g.getNets()),
        [&](GNode n) {
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              int part = g.getData(cc).getPart();
              int nodep;
              if (part == 0) 
                nodep = g.getData(n).p1;
              else 
                nodep = g.getData(n).p2;
              if (nodep == 1) 
                  galois::atomicAdd(g.getData(cc).FS, 1);
               
              if (nodep == (g.getData(n).p1 + g.getData(n).p2))
                  galois::atomicAdd(g.getData(cc).TE, 1);
            }
        },
        galois::loopname("firstFETS"));    
}

void updateGain(GGraph& g, GNode s){
	
  std::array<GNodeBag, 100> cells;
  galois::do_all(galois::iterate(g.getNets()),
    [&](GNode n) {
        auto edge = g.findEdge(n, s);
        if(edge != g.edge_end(n)) {
            g.getData(n, galois::MethodFlag::UNPROTECTED).p1 = 0;
            g.getData(n, galois::MethodFlag::UNPROTECTED).p2 = 0;
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
	      int idx = g.getData(cc, galois::MethodFlag::UNPROTECTED).nodeid%100;
              cells[idx].push(cc);
	      int part = g.getData(cc, galois::MethodFlag::UNPROTECTED).getPart();
              if (part == 0) g.getData(n).p1++;
              else g.getData(n).p2++;
            }
        }
    },
  galois::steal(), galois::loopname("initGain"));
  std::vector<GNode> cell[100];

  galois::do_all(galois::iterate(cells),
  [&](GNodeBag& b){
      if (b.begin() == b.end()) return;
      GNode vv = *b.begin();
      int idx = g.getData(vv).nodeid % 100;
      for (auto v : b) {
          cell[idx].push_back(v);
          g.getData(v).FS.store(0);
          g.getData(v).TE.store(0);
      }
        },
        galois::steal(), galois::loopname("initGain"));


  galois::do_all(galois::iterate(g.getNets()),
        [&](GNode n) {
            for (auto x : g.edges(n)) {
                auto cc = g.getEdgeDst(x);
		int idx = g.getData(cc, galois::MethodFlag::UNPROTECTED).nodeid%100;
		if(std::find(cell[idx].begin(), cell[idx].end(),cc) != cell[idx].end()){
                    int part = g.getData(cc, galois::MethodFlag::UNPROTECTED).getPart();
                    int nodep;
                    if (part == 0)
                        nodep = g.getData(n).p1;
                    else
                        nodep = g.getData(n).p2;
                    if (nodep == 1)
                        galois::atomicAdd(g.getData(cc).FS, 1);
                    if (nodep == (g.getData(n).p1 + g.getData(n).p2))
                        galois::atomicAdd(g.getData(cc).TE, 1);
                }
            }
        },
        galois::steal(), galois::loopname("FETS"));
}

} // namespace

void partition(MetisGraph* mcg) {
  //std::cout<<"in partition\n";
   unsigned newSize = std::distance(mcg->getGraph()->cellList().begin(), mcg->getGraph()->cellList().end());
  //std::cout<<newSize<<"\n";
  //std::vector<partInfo> parts(numPartitions);
  //assert(fineMetisGraphWeight == mcg->getTotalWeight());
  //parts[0] = partInfo(fineMetisGraphWeight);
  GGraph* g = mcg->getGraph();
  galois::GAccumulator<unsigned int> accum;
  int waccum;
  galois::GAccumulator<unsigned int> accumZ;
  GNodeBag nodelist;
  galois::do_all(
      galois::iterate(mcg->getGraph()->cellList()),
      [&](GNode item) {
        accum += g->getData(item).getWeight();
        g->getData(item, galois::MethodFlag::UNPROTECTED).initRefine(1, true);
        g->getData(item, galois::MethodFlag::UNPROTECTED).initPartition();
       // g->getData(item, galois::MethodFlag::UNPROTECTED).FS.store(0);
       // g->getData(item, galois::MethodFlag::UNPROTECTED).TE.store(0);
      },
      galois::loopname("initPart"));

  galois::do_all(
      galois::iterate(g->getNets()),
      [&](GNode item) {
        for (auto c : g->edges(item)) {
          auto n = g->getEdgeDst(c);
          g->getData(n).setPart(0);
        }
      },
      galois::loopname("initones")); 
  GNodeBag nodelistoz;
  galois::do_all(
      galois::iterate(g->cellList()),
      [&](GNode item) {
        if (g->getData(item).getPart() == 0) { 
           accumZ += g->getData(item).getWeight();
           nodelist.push(item);
        }
        else nodelistoz.push(item);
        
      },
      galois::loopname("initones")); 
  unsigned newWeight = 0;
  waccum = accum.reduce() - accumZ.reduce();
  unsigned targetWeight = accum.reduce() / 2;

  if (accumZ.reduce() > waccum) {
  int gain = waccum;
  initGain(*g);
  while(1) {
  //initGain(*g);
    std::vector<GNode> nodeListz;
    GNodeBag nodelistz;
    galois::do_all(
      galois::iterate(nodelist),
      [&](GNode node) {
    //for (auto node : nodelist) {
      unsigned pp = g->getData(node).getPart();
      if (pp == 0) {
        nodelistz.push(node);
      }        
    },	
      galois::loopname("while")); 
    for (auto c :nodelistz) nodeListz.push_back(c);
    std::sort(nodeListz.begin(), nodeListz.end(), [&g] (GNode& lpw, GNode& rpw) {
      return (float) ((g->getData(lpw).getGains()) * (1.0f / g->getData(lpw).getWeight())) > (float)((g->getData(rpw).getGains()) * (1.0f / g->getData(rpw).getWeight()));
    });
    int i = 0;
 //   for (auto zz : nodeListz) {
    auto zz = *nodeListz.begin();
    g->getData(zz).setPart(1);
    gain += g->getData(zz).getWeight();
    //std::cout<<" weight "<<g->getData(zz).getWeight()<<"\n";
    
    i++;
   // if (gain >= targetWeight) break;
   //if(i > sqrt(newSize)) break;
  //}
	
    if (gain >= targetWeight) break;
    updateGain(*g,zz);

  }

}
else {
  
  int gain = accumZ.reduce();
 // std::cout<<"gain is "<<gain<<"\n";
  initGain(*g);
  while(1) {
  //initGain(*g);
    std::vector<GNode> nodeListz;
    GNodeBag nodelistz;
    galois::do_all(
      galois::iterate(nodelistoz),
      [&](GNode node) {
    //for (auto node : nodelist) {
      unsigned pp = g->getData(node).getPart();
      if (pp == 1) {
        nodelistz.push(node);
      }        
    },	
      galois::loopname("while")); 
    for (auto c :nodelistz) nodeListz.push_back(c);
	
    std::sort(nodeListz.begin(), nodeListz.end(), [&g] (GNode& lpw, GNode& rpw) {
      return (float) ((g->getData(lpw).getGains()) * (1.0f / g->getData(lpw).getWeight())) > (float)((g->getData(rpw).getGains()) * (1.0f / g->getData(rpw).getWeight()));
    });

  int i = 0;
//  for (auto zz : nodeListz) {
  auto zz = *nodeListz.begin();
  g->getData(zz).setPart(0);
    //std::cout<<" weight "<<g->getData(zz).getWeight()<<"\n";
  gain += g->getData(zz).getWeight();
    
    i++;
   // if (gain >= targetWeight) break;
   // if(i > sqrt(newSize)) break;
 // }
	
   if (gain >= targetWeight) break;

   updateGain(*g,zz);
  }
}
 // std::cout<<"after partition\n";
/*  int count = 0;
  int zero = 0;
  for (auto c : mcg->getGraph()->cellList()) {
    if (mcg->getGraph()->getData(c).getPart() == 1) {
        count += mcg->getGraph()->getData(c).getWeight();
    }
    else zero += mcg->getGraph()->getData(c).getWeight();
  }
  std::cout<<" number of ones " <<count<<"and number of zeros "<<zero<<"\n";*/
  
}
