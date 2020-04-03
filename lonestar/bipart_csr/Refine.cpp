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
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "bipart.h"
#include "galois/AtomicHelpers.h"
#include <set>
#include <iostream>
#include <fstream>

namespace {

// This is only used on the terminal graph (find graph)
// Should workd for hmetis

/*int calculate_cutsize(GGraph& g) {

  GNodeBag bag;
  galois::do_all(galois::iterate(g.getNets()),
        [&](GNode n) {
            auto c = g.edges(n).begin();
            GNode cn = g.getEdgeDst(*c);
            int part = g.getData(cn).getPart();
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              int partc = g.getData(cc).getPart();
              if (partc != part) {
                bag.push(n);
                return;
              }
            }
        },
        galois::loopname("cutsize"));
  return std::distance(bag.begin(), bag.end());
}*/

/*int calculate_cutsize(GGraph& g, std::map<GNode, unsigned> part) {

  GNodeBag bag;
  galois::do_all(galois::iterate(g.getNets()),
        [&](GNode n) {
            auto c = g.edges(n).begin();
            GNode cn = g.getEdgeDst(*c);
            unsigned ppart = part[cn];
            for (auto x : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
              auto cc = g.getEdgeDst(x);
              unsigned partc = part[cc];
              if (partc != ppart) {
                bag.push(n);
                return;
              }
            }
        },
        galois::steal(), galois::loopname("cutsize"));
  return std::distance(bag.begin(), bag.end());
}*/
void projectPart(MetisGraph* Graph) {
  GGraph* fineGraph   = Graph->getFinerGraph()->getGraph();
  GGraph* coarseGraph = Graph->getGraph();
  galois::do_all(galois::iterate(*fineGraph),
                 [&](GNode n) {
                   if (n < fineGraph->hedges) return;
                   auto parent = fineGraph->getData(n).getParent();
                   auto& cn      = coarseGraph->getData(parent);
                   unsigned part = cn.getPart();
                   fineGraph->getData(n).setPart(part);
                 },
                 galois::loopname("project"));
}



void initGains(GGraph& g, int pass) {
  std::string name = "initgain";
  std::string fetsref = "FETSREF_";// + std::to_string(pass);

  galois::do_all(galois::iterate(g),
        [&](GNode n) {
              if (n <g.hedges ) return;
              g.getData(n).FS.store(0);
              g.getData(n).TE.store(0);
        },
        galois::loopname(name.c_str()));
  galois::InsertBag<std::pair<GNode, GGraph::edge_iterator> >bag;
  typedef std::map<GNode, int> mapTy;
  typedef galois::substrate::PerThreadStorage<mapTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  galois::do_all(galois::iterate(g),
        [&](GNode n) {
        if (g.hedges <= n) return;
        auto& edges = *edgesThreadLocal.getLocal();
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
        galois::loopname("initGains"));    
}

void unlock(GGraph& g) {
    galois::do_all(galois::iterate(g),
                [&](GNode n) {
    if (n < g.hedges) return;
    g.getData(n).counter = 0;
  },
  galois::loopname("unlock"));

}

void unlocked(GGraph& g) {
    galois::do_all(galois::iterate(g),
                [&](GNode n) {
    if (n < g.hedges) return;
    g.getData(n).setLocked(false);
  },
  galois::loopname("unlocked"));

}
//refine
void parallel_refine_KF(GGraph& g, float tol, unsigned refineTo) {

  //std::cout<<"in parallel balance\n";
  typedef galois::gstl::Vector<unsigned> VecTy;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  std::string name = "findZandO";

  //typedef galois::worklists::PerSocketChunkFIFO<8> Chunk;
  unsigned Size = g.hnodes;//std::distance(g.cellList().begin(), g.cellList().end());

  galois::GAccumulator<unsigned int> accum;
  galois::GAccumulator<unsigned int> nodeSize;
  galois::do_all(galois::iterate(g), 
  [&](GNode n) {
     if (n < g.hedges) return;
     nodeSize += g.getData(n).getWeight();
     if (g.getData(n).getPart() > 0)
       accum += g.getData(n).getWeight();
  },
  galois::loopname("make balance"));
  //std::cout<<"weight of 0 : "<< nodeSize.reduce() - accum.reduce()<<"  1: "<<accum.reduce()<<"\n";
  const int hi = (1 + tol) * Size / (2 + tol);
  const int lo = Size - hi;
  int bal = accum.reduce();
  int pass = 0;
  int changed = 0;
 // std::cout<<"cut parallel "<<calculate_cutsize(g)<<"\n";
 // initGain(g);
  while (pass < refineTo) {
  //T.start();
    initGains(g, refineTo);
  //T.stop();
  //std::cout<<"init gain time "<<T.get()<<" for round "<<pass<<"\n";
    GNodeBag nodelistz;
    GNodeBag nodelisto;
    unsigned zeroW = 0;
    unsigned oneW = 0;
    galois::do_all(galois::iterate(g), 
      [&](GNode n) {
          if (n < g.hedges) return;
          if (g.getData(n).FS == 0 && g.getData(n).TE == 0) return;
          int gain = g.getData(n).getGain();
          if (gain < 0) {
            return;
          }
          unsigned pp = g.getData(n).getPart();
          if (pp == 0) {
            nodelistz.push(n);
          }
          else {
            nodelisto.push(n);
          }
      },
    galois::loopname("findZandO"));
    zeroW = std::distance(nodelistz.begin(), nodelistz.end());
    oneW = std::distance(nodelisto.begin(), nodelisto.end());
    unsigned z = 0, o = 0;
    GNodeBag bb;
    std::vector<GNode> bbagz;
    std::vector<GNode> bbago;
    int zw = 0;
    int ow = 0;
    unsigned ts = 0;
    for (auto n : nodelistz) bbagz.push_back(n);
    for (auto n : nodelisto) bbago.push_back(n);
    std::sort(bbagz.begin(), bbagz.end(), [&g] (GNode& lpw, GNode& rpw) {
      if (g.getData(lpw).getGain()  == g.getData(rpw).getGain() ) return g.getData(lpw).nodeid < g.getData(rpw).nodeid;
      return g.getData(lpw).getGain() > g.getData(rpw).getGain();
    });
    std::sort(bbago.begin(), bbago.end(), [&g] (GNode& lpw, GNode& rpw) {
      if (g.getData(lpw).getGain() == g.getData(rpw).getGain())  return g.getData(lpw).nodeid < g.getData(rpw).nodeid;
      return g.getData(lpw).getGain() > g.getData(rpw).getGain();
    });
    if (zeroW <= oneW) {
      for (int i = 0; i < zeroW; i++) {
        bb.push(bbago[i]);
        bb.push(bbagz[i]);
    //    if (i >= sqrt(Size)) break;
      }
      galois::do_all(galois::iterate(bb),
              [&](GNode n) {
                  if (g.getData(n).getPart() == 0)
                     g.getData(n).setPart(1);
                  else 
                     g.getData(n).setPart(0);
                  g.getData(n).counter++;
              },
      galois::loopname("swap"));
   }
   else {
      for (int i = 0; i < oneW; i++) {
        bb.push(bbago[i]);
        bb.push(bbagz[i]);
   //     if (i >= sqrt(Size)) break;
      }
      galois::do_all(galois::iterate(bb),
              [&](GNode n) {
                  if (g.getData(n).getPart() == 0)
                     g.getData(n).setPart(1);
                  else 
                     g.getData(n).setPart(0);
                  g.getData(n).counter++;
              },
      galois::loopname("swap"));
   }
  pass++; 
 }
 unlock(g); 
}
// find the boundary in parallel 
// sort the boundary in parallel
// swap in parallel using for_each (find the smallest and go over that)
unsigned hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}


void parallel_make_balance(GGraph& g, float tol, int p) {

  unsigned Size = g.hnodes;//std::distance(g.cellList().begin(), g.cellList().end());

  galois::GAccumulator<unsigned int> accum;
  galois::GAccumulator<unsigned int> nodeSize;
  galois::do_all(galois::iterate(g),
  [&](GNode n) {
     if (n < g.hedges) return;
     nodeSize += g.getData(n).getWeight();
     if (g.getData(n).getPart() > 0)
       accum += g.getData(n).getWeight();
  },
  galois::loopname("make balance"));

  const int hi = (1 + tol) * nodeSize.reduce() / (2 + tol);
  const int lo = nodeSize.reduce() - hi;
  int bal = accum.reduce();

  int zero_weight = nodeSize.reduce() - accum.reduce();
  int one_weights = bal;
  int pass = 0;
  while(1) {
    if(bal >= lo && bal <= hi)  break;
    initGains(g, p);
	
    //creating buckets
    std::array<std::vector<GNode>, 101> nodeListz;
    std::array<std::vector<GNode>, 101> nodeListo;

    std::array<GNodeBag, 101> nodelistz;
    std::array<GNodeBag, 101> nodelisto;
    
    //bucket for nodes with gan by weight ratio <= -9.0f
    std::vector<GNode> nodeListzNegGain;
    std::vector<GNode> nodeListoNegGain;
    
    GNodeBag nodelistzNegGain;
    GNodeBag nodelistoNegGain;
     
    if (bal < lo) {
	
	//placing each node in an appropriate bucket using the gain by weight ratio
	galois::do_all(galois::iterate(g),
      [&](GNode n) {
          if ( n < g.hedges) return;

          float  gain = ((float) g.getData(n).getGain())/ ((float) g.getData(n).getWeight());
          unsigned pp = g.getData(n).getPart();
          if (pp == 0) {
	    //nodes with gain >= 1.0f are in one bucket
	    if(gain >= 1.0f){
		nodelistz[0].push(n);	
	    }
	    else if(gain >= 0.0f){
	        int d = gain*10.0f;
		int idx = 10 - d;
		nodelistz[idx].push(n);
	    }
	    else if(gain >= -9.0f){
		int d = gain*10.0f - 1;
		int idx = 10 - d;
		nodelistz[idx].push(n);
	    }
	    else{	//NODES with gain by weight ratio <= -9.0f are in one bucket
            	nodelistzNegGain.push(n);
	    }
          }
      }, galois::steal());

	//sorting each bucket in parallel
       galois::do_all(galois::iterate(nodelistz),
      [&](GNodeBag& b) {
		if(b.begin() == b.end()) return;
		
		GNode n = *b.begin();
		float  gain = ((float) g.getData(n).getGain())/ ((float) g.getData(n).getWeight());
		int idx;
		if(gain >= 1.0f)
			idx = 0;
		else if(gain >= 0.0f){
			int d = gain*10.0f;
                	idx = 10 - d;
		}
		else{
			int d = gain*10.0f - 1;
                	idx = 10 - d;
		}
		for (auto x:b){
			nodeListz[idx].push_back(x);
		}

		std::sort(nodeListz[idx].begin(), nodeListz[idx].end(), [&g] (GNode& lpw, GNode& rpw) {
    if (fabs((float)((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) - (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()))) < 0.00001f) return (float)g.getData(lpw).nodeid < (float)g.getData(rpw).nodeid;
      return (float) ((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) > (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()));
    });

	}, galois::steal());

	int i = 0;
	int j = 0;

	//now moving nodes from partition 0 to 1
	while(j <= 100){
		if(nodeListz[j].size() == 0){
			j++;
			continue;
		}

		for(auto zz: nodeListz[j]){
			g.getData(zz).setPart(1);
       			bal += g.getData(zz).getWeight();
        		if(bal >= lo) break;
        		i++;
        		if (i > sqrt(Size)) break;
		}
		if(bal >= lo) break;
		if (i > sqrt(Size)) break;
		j++;
	}

	if(bal >= lo) break;
	if (i > sqrt(Size)) continue;

	//moving nodes from nodeListzNegGain
	//
	if(nodelistzNegGain.begin() == nodelistzNegGain.end())	continue;

	for(auto x: nodelistzNegGain)
		nodeListzNegGain.push_back(x);

	std::sort(nodeListzNegGain.begin(), nodeListzNegGain.end(), [&g] (GNode& lpw, GNode& rpw) {
    if (fabs((float)((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) - (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()))) < 0.00001f) return (float)g.getData(lpw).nodeid < (float)g.getData(rpw).nodeid;
      return (float) ((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) > (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()));
    });

	for(auto zz: nodeListzNegGain){
                        g.getData(zz).setPart(1);
                        bal += g.getData(zz).getWeight();
                        if(bal >= lo) break;
                        i++;
                        if (i > sqrt(Size)) break;
        }
	
	if(bal >= lo) break;

    }//end if

    else {

	//placing each node in an appropriate bucket using the gain by weight ratio
    	galois::do_all(galois::iterate(g),
      [&](GNode n) {
          if (n < g.hedges) return;

          float  gain = ((float) g.getData(n).getGain())/ ((float) g.getData(n).getWeight());
          unsigned pp = g.getData(n).getPart();
          if (pp == 1) {
		//nodes with gain >= 1.0f are in one bucket
		if(gain >= 1.0f){
                nodelisto[0].push(n);
            	}
            	else if(gain >= 0.0f){
                	int d = gain*10.0f;
                	int idx = 10 - d;
                	nodelisto[idx].push(n);
            	}
            	else if(gain > -9.0f){
                	int d = gain*10.0f - 1;
                	int idx = 10 - d;
                	nodelisto[idx].push(n);
            	}
            	else{	//NODES with gain by weight ratio <= -9.0f are in one bucket
		nodelistoNegGain.push(n);
            }
          }
      });     		
		
	//sorting each bucket in parallel
	galois::do_all(galois::iterate(nodelisto),
      [&](GNodeBag& b) {
                if(b.begin() == b.end()) return;

                GNode n = *b.begin();
                float  gain = ((float) g.getData(n).getGain())/ ((float) g.getData(n).getWeight());
                int idx;
                if(gain >= 1.0f)
                        idx = 0;
                else if(gain >= 0.0f){
                        int d = gain*10.0f;
                        idx = 10 - d;
                }
                else{
                        int d = gain*10.0f - 1;
                        idx = 10 - d;
                }
                for (auto x:b){
                        nodeListo[idx].push_back(x);
                }

                std::sort(nodeListo[idx].begin(), nodeListo[idx].end(), [&g] (GNode& lpw, GNode& rpw) {
    if (fabs((float)((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) - (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()))) < 0.00001f) return (float)g.getData(lpw).nodeid < (float)g.getData(rpw).nodeid;
      return (float) ((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) > (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()));
    });

        });

        int i = 0;
        int j = 0;	


	//now moving nodes from partition 1 to 0
	while(j <= 100){
                if(nodeListo[j].size() == 0){
                        j++;
                        continue;
                }

                for(auto zz: nodeListo[j]){
                        g.getData(zz).setPart(0);
                        bal -= g.getData(zz).getWeight();
                        if(bal <= hi) break;
                        i++;
                        if (i > sqrt(Size)) break;
                }
                if(bal <= hi) break;
                if (i > sqrt(Size)) break;
                j++;
        }

        if(bal <= hi) break;
        if (i > sqrt(Size)) continue;


	//moving nodes from nodeListoNegGain
	//        
	 if(nodelistoNegGain.begin() == nodelistoNegGain.end())  continue;

        for(auto x: nodelistoNegGain)
                nodeListoNegGain.push_back(x);

        std::sort(nodeListoNegGain.begin(), nodeListoNegGain.end(), [&g] (GNode& lpw, GNode& rpw) {
    if (fabs((float)((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) - (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()))) < 0.00001f) return (float)g.getData(lpw).nodeid < (float)g.getData(rpw).nodeid;
      return (float) ((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) > (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()));
    });

        for(auto zz: nodeListoNegGain){
                        g.getData(zz).setPart(0);
                        bal -= g.getData(zz).getWeight();
                        if(bal <= hi) break;
                        i++;
                        if (i > sqrt(Size)) break;
        }

        if(bal <= hi) break;
    } //end else
	
  }//end while
}
void make_balance(GGraph& g, float tol, int p) {

 unsigned Size = g.hnodes;//std::distance(g.cellList().begin(), g.cellList().end());

  galois::GAccumulator<unsigned int> accum;
  galois::GAccumulator<unsigned int> nodeSize;
  galois::do_all(galois::iterate(g),
  [&](GNode n) {
     if (n < g.hedges) return;
     nodeSize += g.getData(n).getWeight();
     if (g.getData(n).getPart() > 0)
       accum += g.getData(n).getWeight();
  },
  galois::loopname("make balance"));
  //std::cout<<"weight of 0 : "<< nodeSize.reduce() - accum.reduce()<<"  1: "<<accum.reduce()<<"\n";
  const int hi = (1 + tol) * nodeSize.reduce() / (2 + tol);
  const int lo = nodeSize.reduce() - hi;
  int bal = accum.reduce();

  int zero_weight = nodeSize.reduce() - accum.reduce();
  int one_weights = bal;
  int pass = 0;
  while(1) {
    if(bal >= lo && bal <= hi)	break;
    initGains(g, p);
    std::vector<GNode> nodeListz;
    std::vector<GNode> nodeListo;
    std::set<GNode> nodelistz;
    std::set<GNode> nodelisto;
    if (bal < lo) {
      for (int b = 0; b < g.hedges; b++) {
        for (auto n : g.edges(b)) {
          auto node = g.getEdgeDst(n);
          unsigned pp = g.getData(node).getPart();
          if (pp == 0) {
            nodelistz.insert(node);
          }
      	}
    }
		
    for (auto x : nodelistz)
       nodeListz.push_back(x);
    std::sort(nodeListz.begin(), nodeListz.end(), [&g] (GNode& lpw, GNode& rpw) {
    if (fabs((float)((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) - (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()))) < 0.00001f) return (float)g.getData(lpw).nodeid < (float)g.getData(rpw).nodeid;
      return (float) ((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) > (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()));
    });
    int i = 0;
      for (auto zz : nodeListz) {
       // if (g.getData(zz).counter > 2)continue;
//        g.getData(zz).counter++;
        g.getData(zz).setPart(1);
	bal += g.getData(zz).getWeight();
	if(bal >= lo) break;
	i++;
	if (i > sqrt(Size)) break;
      }
      if(bal >= lo) break;//continue;
     
    }
	
    else {
      for (GNode b = 0; b < g.hedges; b++) {
        for (auto n : g.edges(b)) {
          auto node = g.getEdgeDst(n);
          unsigned pp = g.getData(node).getPart();
          if (pp == 1) {
            nodelisto.insert(node);
          }

        }
      }

      for (auto x : nodelisto)
        nodeListo.push_back(x);

      std::sort(nodeListo.begin(), nodeListo.end(), [&g] (GNode& lpw, GNode& rpw) {
    if (fabs((float)((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) - (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()))) < 0.00001f) return (float)g.getData(lpw).nodeid < (float)g.getData(rpw).nodeid;
      return (float) ((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) > (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()));
    });

      int i = 0;
      for (auto zz : nodeListo) {
    //    if (g.getData(zz).counter > 20)continue;
        g.getData(zz).setPart(0);
      //  g.getData(zz).counter++;
        bal -= g.getData(zz).getWeight();
        if(bal <= hi) break;
        i++;
        if (i > sqrt(Size)) break;
      }

      if (bal <= hi) break;//continue;
    }
  pass++;
  }
}

} // namespace

void refine(MetisGraph* coarseGraph, unsigned refineTo) {
  MetisGraph* fineG;
  const float ratio = 55.0 / 45.0;  // change if needed
  const float tol = std::max(ratio, 1 - ratio) - 1;
  do {
    fineG = coarseGraph;
    MetisGraph* fineGraph = coarseGraph->getFinerGraph();
    auto gg = coarseGraph->getGraph();

    parallel_refine_KF(*gg, tol, refineTo);
    parallel_make_balance(*gg, tol, 2);

    bool do_pro = true;
    if (fineGraph && do_pro) {
      projectPart(coarseGraph);
    }
   } while ((coarseGraph = coarseGraph->getFinerGraph()));
}

