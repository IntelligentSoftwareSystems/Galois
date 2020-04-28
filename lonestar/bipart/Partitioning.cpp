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
#include "bipart.h"
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

namespace {
// final
__attribute__((unused)) int cut(GGraph& g) {

  GNodeBag bag;
  galois::do_all(galois::iterate(g),
        [&](GNode n) {
          if (g.hedges <= n) return;
          for (auto cell : g.edges(n)) {
            auto c = g.getEdgeDst(cell);
            int part = g.getData(c).getPart();
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              int partc = g.getData(cc).getPart();
              if (partc != part) {
                bag.push(n);
                return;
              }

            }
          }
        },
        galois::loopname("cutsize"));
  return std::distance(bag.begin(), bag.end());
}

void initGain(GGraph& g) {
  galois::do_all(galois::iterate(g),
        [&](GNode n) {
            if (n < g.hedges) return;
            g.getData(n).FS.store(0);
            g.getData(n).TE.store(0);
        },
        galois::loopname("firstinit"));

  typedef std::map<GNode, int> mapTy;
  typedef galois::substrate::PerThreadStorage<mapTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  galois::do_all(galois::iterate(g),
        [&](GNode n) {
        if (g.hedges <= n) return; 
           int p1=0;
						int p2 = 0;
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              int part = g.getData(cc).getPart();
              if (part == 0) p1++;
              else p2++;
            if (p1 > 1 && p2 > 1) break;
            }
            if (!(p1 > 1 && p2 > 1) && (p1 + p2 > 1) ) {
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              int part = g.getData(cc).getPart();
              int nodep;
              if (part == 0) 
                nodep = p1;
              else 
                nodep = p2;
              if (nodep == 1) {
                  galois::atomicAdd(g.getData(cc).FS, 1);
              }
              if (nodep == (p1 + p2)) {
                  galois::atomicAdd(g.getData(cc).TE, 1);
		}
            }
	}
        },
	galois::steal(),
        galois::loopname("initGainsPart"));    
}

} // namespace

// Final
void partition(MetisGraph* mcg) {
  GGraph* g = mcg->getGraph();
  galois::GAccumulator<unsigned int> accum;
  int waccum;
  galois::GAccumulator<unsigned int> accumZ;
  GNodeBag nodelist;
  galois::do_all(
      galois::iterate(g->hedges, g->size()),
      [&](GNode item) {
        accum += g->getData(item).getWeight();
        g->getData(item, galois::MethodFlag::UNPROTECTED).initRefine(1, true);
        g->getData(item, galois::MethodFlag::UNPROTECTED).initPartition();
      },
      galois::loopname("initPart"));

  galois::do_all(galois::iterate(size_t{0}, g->hedges),
      [&](GNode item) {
        for (auto c : g->edges(item)) {
          auto n = g->getEdgeDst(c);
          g->getData(n).setPart(0);
        }
      },
      galois::loopname("initones")); 
  GNodeBag nodelistoz;
  galois::do_all(
      galois::iterate(g->hedges, g->size()),
      [&](GNode item) {
        if (g->getData(item).getPart() == 0) { 
           accumZ += g->getData(item).getWeight();
           nodelist.push(item);
        }
        else nodelistoz.push(item);
        
      },
      galois::loopname("initones")); 
  unsigned newSize = accum.reduce();
  waccum = accum.reduce() - accumZ.reduce();
  unsigned targetWeight = accum.reduce() / 2;

  if (static_cast<long>(accumZ.reduce()) > waccum) {
  int gain = waccum;
  //initGain(*g);
  while(1) {
  initGain(*g);
    std::vector<GNode> nodeListz;
    GNodeBag nodelistz;
    galois::do_all(
      galois::iterate(nodelist),
      [&](GNode node) {
      unsigned pp = g->getData(node).getPart();
      if (pp == 0) {
        nodelistz.push(node);
      }        
    },	
      galois::loopname("while")); 

    for (auto c :nodelistz) nodeListz.push_back(c);
    std::sort(nodeListz.begin(), nodeListz.end(), [&g] (GNode& lpw, GNode& rpw) {
    if (fabs((float)((g->getData(lpw).getGain()) * (1.0f / g->getData(lpw).getWeight())) - (float)((g->getData(rpw).getGain()) * (1.0f / g->getData(rpw).getWeight()))) < 0.00001f) return (float)g->getData(lpw).nodeid < (float)g->getData(rpw).nodeid;
      return (float) ((g->getData(lpw).getGain()) * (1.0f / g->getData(lpw).getWeight())) > (float)((g->getData(rpw).getGain()) * (1.0f / g->getData(rpw).getWeight()));
    });
    int i = 0;
    for (auto zz : nodeListz) {
    //auto zz = *nodeListz.begin();
    g->getData(zz).setPart(1);
    gain += g->getData(zz).getWeight();
    //std::cout<<" weight "<<g->getData(zz).getWeight()<<"\n";
    
    i++;
    if (gain >= static_cast<long>(targetWeight)) break;
   if(i > sqrt(newSize)) break;
  }
	
    if (gain >= static_cast<long>(targetWeight)) break;
    //updateGain(*g,zz);

  }

}
else {
  
  int gain = accumZ.reduce();
 // std::cout<<"gain is "<<gain<<"\n";
  //initGain(*g);
  while(1) {
  initGain(*g);
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
    if (fabs((float)((g->getData(lpw).getGain()) * (1.0f / g->getData(lpw).getWeight())) - (float)((g->getData(rpw).getGain()) * (1.0f / g->getData(rpw).getWeight()))) < 0.00001f) return (float)g->getData(lpw).nodeid < (float)g->getData(rpw).nodeid;
      return (float) ((g->getData(lpw).getGain()) * (1.0f / g->getData(lpw).getWeight())) > (float)((g->getData(rpw).getGain()) * (1.0f / g->getData(rpw).getWeight()));
    });

  int i = 0;
  for (auto zz : nodeListz) {
  //auto zz = *nodeListz.begin();
  g->getData(zz).setPart(0);
  gain += g->getData(zz).getWeight();
    
    i++;
    if (gain >= static_cast<long>(targetWeight)) break;
    if(i > sqrt(newSize)) break;
  }
	
   if (gain >= static_cast<long>(targetWeight)) break;

   //updateGain(*g,zz);
  }
}
  
}

