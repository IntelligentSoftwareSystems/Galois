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
#include "Metis.h"
#include "galois/AtomicHelpers.h"
#include <set>
#include <iostream>
#include <fstream>

namespace {

// This is only used on the terminal graph (find graph)
// Should workd for hmetis
bool isBoundary(GGraph& g, GNode n) {
  auto c = g.edges(n).begin();
  GNode dst = g.getEdgeDst(*c);
  unsigned int nPart = g.getData(dst).getPart();
  for (auto jj : g.edges(n)) 
    if (g.getData(g.getEdgeDst(jj)).getPart() != nPart)
      return true;
    
  return false;
}

void findBoundary(GNodeBag& bag, GGraph& cg) {

  galois::do_all(galois::iterate(cg.getNets()),
                 [&](GNode n) {
                   if (isBoundary(cg, n))
                     bag.push(n);
                 },
                 galois::loopname("findBoundary"));
}

int calculate_cutsize(GGraph& g) {

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
}

int calculate_cutsize(GGraph& g, std::map<GNode, unsigned> part) {

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
}
void projectPart(MetisGraph* Graph) {
  GGraph* fineGraph   = Graph->getFinerGraph()->getGraph();
  GGraph* coarseGraph = Graph->getGraph();
  galois::do_all(galois::iterate(fineGraph->cellList()),
                 [&](GNode n) {
                   auto parent = fineGraph->getData(n).getParent();
                   auto& cn      = coarseGraph->getData(parent);
                   unsigned part = cn.getPart();
                   fineGraph->getData(n).setPart(part);
                 },
                 galois::loopname("project"));
}

void initGain(GGraph& g, int pass) {
  std::string name = "initgain";// + std::to_string(pass);
  std::string fetsref = "FETSREF_";// + std::to_string(pass);

  galois::do_all(galois::iterate(g.cellList()),
        [&](GNode n) {

		g.getData(n).gains = 0;
        },
        galois::loopname(name.c_str()));

  //int MAX_SIZE = 11000000;
  typedef std::array<int, 2000000> arrayTy;
  typedef galois::substrate::PerThreadStorage<arrayTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;

  auto& edges = *edgesThreadLocal.getLocal();
  //edges.fill(0);
 
  galois::do_all(galois::iterate(g.getNets()),
        [&](GNode n) {
        auto& edges = *edgesThreadLocal.getLocal();
            g.getData(n).p1 = 0;
            g.getData(n).p2 = 0;
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              int part = g.getData(cc).getPart();
              if (part == 0) g.getData(n).p1++;
              else g.getData(n).p2++;
            if (g.getData(n).p1 > 1 && g.getData(n).p2 > 1) break;
            }
            if (!(g.getData(n).p1 > 1 && g.getData(n).p2 > 1)) {
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              int part = g.getData(cc).getPart();
              int nodep;
              if (part == 0)
                nodep = g.getData(n).p1;
              else
                nodep = g.getData(n).p2;
              if (nodep == 1)
		edges[g.getData(cc).nodeid]++;
	      if (nodep == (g.getData(n).p1 + g.getData(n).p2))
		edges[g.getData(cc).nodeid]--;
	    }
	    }
	}, galois::loopname(fetsref.c_str()));

  galois::do_all(galois::iterate(g.cellList()),
        [&](GNode n) {
        int gains = 0;
	for (int i = 0; i < edgesThreadLocal.size(); i++) {
      		auto& edges = *edgesThreadLocal.getRemote(i);
		gains += edges[g.getData(n).nodeid];
	}
        g.getData(n).gains = gains;
    },
    galois::loopname("edges"));

}

void initGainss(GGraph& g, int pass) {
  std::string name = "initgain";// + std::to_string(pass);
  std::string fetsref = "FETSREF_";// + std::to_string(pass);

  galois::do_all(galois::iterate(g.cellList()),
        [&](GNode n) {
              g.getData(n).FS.store(0);
              g.getData(n).TE.store(0);
             //g.getData(n).gains = 0;
        },
        galois::loopname(name.c_str()));

  typedef std::map<GNode, int> mapTy;
  typedef galois::substrate::PerThreadStorage<mapTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  galois::do_all(galois::iterate(g.getNets()),
        [&](GNode n) {
        auto& edges = *edgesThreadLocal.getLocal();
            g.getData(n).p1 = 0;
            g.getData(n).p2 = 0;
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              int part = g.getData(cc).getPart();
              if (part == 0) g.getData(n).p1++;
              else g.getData(n).p2++;
            if (g.getData(n).p1 > 1 && g.getData(n).p2 > 1) break;
            }
            if (!(g.getData(n).p1 > 1 && g.getData(n).p2 > 1)) {
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              int part = g.getData(cc).getPart();
              int nodep;
              if (part == 0) 
                nodep = g.getData(n).p1;
              else 
                nodep = g.getData(n).p2;
              if (nodep == 1) {
                  galois::atomicAdd(g.getData(cc).FS, 1);
           //     if (edges.find(cc) != edges.end())
             //     edges[cc] += 1;
              //  else edges[cc] = 1;
              }
              if (nodep == (g.getData(n).p1 + g.getData(n).p2)) {
                //if (edges.find(cc) != edges.end())
                 // edges[cc] -= 1;
                //else edges[cc] = -1;
                  galois::atomicAdd(g.getData(cc).TE, 1);
		}
            }
	}
        },
	galois::steal(),
        galois::loopname(fetsref.c_str()));    

  /*galois::do_all(galois::iterate(g.cellList()),
        [&](GNode n) {
	bool updated = false;
    for (int i = 0; i < edgesThreadLocal.size(); i++) {
      auto& edges = *edgesThreadLocal.getRemote(i);
      if (edges.find(n) != edges.end())
        g.getData(n).gains += edges[n];
	updated = true;
      }
	if (updated) count++;
    },
    galois::loopname("edges"));    
*/
}

void initGains(GGraph& g, int pass) {
  std::string name = "initgain";
  std::string fetsref = "FETSREF_";// + std::to_string(pass);

  /*galois::do_all(galois::iterate(g.cellList()),
        [&](GNode n) {
              g.getData(n).FS.store(0);
              g.getData(n).TE.store(0);
        },
        galois::loopname(name.c_str()));*/
  galois::InsertBag<std::pair<GNode, GGraph::edge_iterator> >bag;
  galois::do_all(galois::iterate(g.getNets()),
        [&](GNode n) {
            g.getData(n).p1 = 0;
            g.getData(n).p2 = 0;
            for (auto x : g.edges(n)) {
              auto cc = g.getEdgeDst(x);
              if (g.getData(cc).FS != 0) {
                g.getData(cc).FS.store(0);
                g.getData(cc).TE.store(0);
              }
              int part = g.getData(cc).getPart();
              if (part == 0) g.getData(n).p1++;
              else g.getData(n).p2++;
            if (g.getData(n).p1 > 1 && g.getData(n).p2 > 1) continue;
              bag.push(std::make_pair(n, x));
            }
        },
	galois::steal(),
        galois::loopname(fetsref.c_str()));    
    galois::do_all(galois::iterate(bag),
                [&](std::pair<GNode, GGraph::edge_iterator> nn) {
              auto n = nn.first;
              auto cc = g.getEdgeDst(nn.second);
              int part = g.getData(cc).getPart();
              int nodep;
              if (part == 0) 
                nodep = g.getData(n).p1;
              else 
                nodep = g.getData(n).p2;
              if (nodep == 1) {
                  galois::atomicAdd(g.getData(cc).FS, 1);
              }
              if (nodep == (g.getData(n).p1 + g.getData(n).p2)) {
                  galois::atomicAdd(g.getData(cc).TE, 1);
		}
 
        },
	galois::steal(),
        galois::loopname("secondFSTE"));    
}

void unlock(GGraph& g) {
    galois::do_all(galois::iterate(g.cellList()),
                [&](GNode n) {
    g.getData(n).counter = 0;
  },
  galois::loopname("unlock"));

}

void unlocked(GGraph& g) {
    galois::do_all(galois::iterate(g.cellList()),
                [&](GNode n) {
    g.getData(n).setLocked(false);
  },
  galois::loopname("unlocked"));

}

void parallel_refine_KF(GGraph& g, float tol, unsigned refineTo) {

  //std::cout<<"in parallel balance\n";
  typedef galois::gstl::Vector<unsigned> VecTy;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  std::string name = "findZandO";

  //typedef galois::worklists::PerSocketChunkFIFO<8> Chunk;
  unsigned Size = std::distance(g.cellList().begin(), g.cellList().end());

  galois::GAccumulator<unsigned int> accum;
  galois::GAccumulator<unsigned int> nodeSize;
  galois::do_all(galois::iterate(g.cellList()), 
  [&](GNode n) {
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
  while (pass < 4) {
  //T.start();
    initGains(g, refineTo);
  //T.stop();
  //std::cout<<"init gain time "<<T.get()<<" for round "<<pass<<"\n";
    GNodeBag nodelistz;
    GNodeBag nodelisto;
    unsigned zeroW = 0;
    unsigned oneW = 0;
    galois::do_all(galois::iterate(g.cellList()), 
      [&](GNode n) {
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
        if (i >= sqrt(Size)) break;
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
        if (i >= sqrt(Size)) break;
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
 // int cutb = calculate_cutsize(g);
  //  std::cout<<"cut size "<<calculate_cutsize(g)<<"\n";
 // if (cut - cutb < 100) break;
  //unlocked(g);
 // updateGain(g, bb);
  pass++; 
 }
 unlock(g); 
}
void refine_KF(GGraph& g, float tol) {

  typedef galois::gstl::Vector<unsigned> VecTy;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;

  typedef galois::worklists::PerSocketChunkFIFO<8> Chunk;
  unsigned Size = std::distance(g.cellList().begin(), g.cellList().end());

  galois::GAccumulator<unsigned int> accum;
  galois::GAccumulator<unsigned int> nodeSize;
  galois::do_all(galois::iterate(g.cellList()), 
  [&](GNode n) {
     nodeSize += g.getData(n).getWeight();
     if (g.getData(n).getPart() > 0)
       accum += g.getData(n).getWeight();
  },
  galois::loopname("make balance"));
  std::cout<<"weight of 0 : "<< nodeSize.reduce() - accum.reduce()<<"  1: "<<accum.reduce()<<"\n";
  const int hi = (1 + tol) * Size / (2 + tol);
  const int lo = Size - hi;
  int bal = accum.reduce();
  int pass = 0;
  int changed = 0;
  std::cout<<"cut  "<<calculate_cutsize(g)<<"\n";
  while (pass < 100) {
  int cut = calculate_cutsize(g);
    initGain(g, -1);
    GNodeBag bag;
    findBoundary(bag, g);
    std::vector<GNode> nodeListz;
    std::vector<GNode> nodeListo;
    std::set<GNode> nodelistz;
    std::set<GNode> nodelisto;
    unsigned zeroW = 0;
    unsigned oneW = 0;
    for (auto b : bag) {
      for (auto n : g.edges(b)) {
        auto node = g.getEdgeDst(n);
        int gainz = g.getData(node).getGain();
        if (gainz < 0) continue;
        unsigned pp = g.getData(node).getPart();
        if (pp == 0) {
          nodelistz.insert(node);
        }
        else {
          nodelisto.insert(node);
        }
      }
    }
    for (auto x : nodelisto) { 
      nodeListo.push_back(x);
      oneW ++;
    }
    for (auto x : nodelistz) {
      nodeListz.push_back(x);
      zeroW ++;
    }
    std::cout<<nodeListo.size()<<" "<<nodeListz.size()<<"\n";

    std::sort(nodeListz.begin(), nodeListz.end(), [&g] (GNode& lpw, GNode& rpw) {
      if (fabs((float)((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) - (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight())) < 0.0001f)) return (float)g.getData(lpw).nodeid < (float)g.getData(rpw).nodeid;
      return (float) ((g.getData(lpw).getGain()) ) > (float)((g.getData(rpw).getGain()) );
    });
    std::sort(nodeListo.begin(), nodeListo.end(), [&g] (GNode& lpw, GNode& rpw) {
      if (fabs((float)(g.getData(lpw).getGain() * (1.0f / g.getData(lpw).getWeight())) - (float)((g.getData(rpw).FS - g.getData(rpw).TE) * (1.0f / g.getData(rpw).getWeight())) < 0.0001f))  return (float)g.getData(lpw).nodeid < (float)g.getData(rpw).nodeid;
      return (float)((g.getData(lpw).getGain()) ) > (float)((g.getData(rpw).getGain()));
    });

    unsigned sizez = nodeListz.size();
    unsigned sizeo = nodeListo.size();
   // std::cout<<"size of z "<<sizez<<" "<<sizeo<<"\n";
    unsigned z = 0, o = 0;
    GNodeBag bbagz, bbago;
    int zw = 0;
    int ow = 0;
    unsigned ts = 0;
    if (zeroW <= oneW) {
      int i = 0;
      for (auto zz : nodeListz) {
        //auto zz = nodeListz.at(i);
        float gainz = (g.getData(zz).getGain());
        if (gainz <= 0.4f) break;
        bbagz.push_back(zz);
        zw++;
        i++;
        if (i > sqrt(Size)) break;
      }
      auto ooo = nodeListo.begin();
      while (ow < zw && ooo != nodeListo.end()) {
        auto oo = *ooo;
        bbago.push_back(oo);
        ow ++;
        ++ooo;
      }
      for (auto b : bbagz)  { 
        g.getData(b).setPart(1);
        g.getData(b).counter++;
      }
      for (auto b : bbago) {
        g.getData(b).setPart(0);
        g.getData(b).counter++;
        ts ++;
        if (ts >= zw) break;
      }
     }
     else {
        int i = 0;
        for (auto oo : nodeListo) {
           float gaino = (g.getData(oo).getGain()) * (1.0f / g.getData(oo).getWeight());
           if (gaino <= 0.4f) break;
           bbago.push_back(oo);
           ow++;
           i++;
           if (i > sqrt(Size)) break;
        }
        unsigned j = 0;
        auto ooo = nodeListz.begin();
        while (zw < ow && ooo != nodeListz.end()) {
           auto oo = *ooo;
           bbagz.push_back(oo);
           zw ++;
           ooo++;
           j++;
        }
      for (auto b : bbago) { 
        g.getData(b).setPart(0);
        g.getData(b).counter++;
      }
      for (auto b : bbagz) {
        g.getData(b).setPart(1);
        g.getData(b).counter++;
        ts++;
        if (ts >= zw) break;
      }
       // ow -> 1 to 0
   } 
    pass++;
    if(ts == 0) break;
  int cutb = calculate_cutsize(g);
    std::cout<<"cut size "<<calculate_cutsize(g)<<"\n";
  if (cut - cutb < 100) break;
 }
 unlock(g); 
}
// find the boundary in parallel 
// sort the boundary in parallel
// swap in parallel using for_each (find the smallest and go over that)
void refine_by_swap(GGraph& g, float tol) {

  typedef galois::gstl::Vector<unsigned> VecTy;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;

  typedef galois::worklists::PerSocketChunkFIFO<8> Chunk;
  unsigned Size = std::distance(g.cellList().begin(), g.cellList().end());

  galois::GAccumulator<unsigned int> accum;
  galois::GAccumulator<unsigned int> nodeSize;
  galois::do_all(galois::iterate(g.cellList()), 
  [&](GNode n) {
     nodeSize += g.getData(n).getWeight();
     if (g.getData(n).getPart() > 0)
       accum += g.getData(n).getWeight();
  },
  galois::loopname("make balance"));
  std::cout<<"weight of 0 : "<< nodeSize.reduce() - accum.reduce()<<"  1: "<<accum.reduce()<<"\n";
  const int hi = (1 + tol) * nodeSize.reduce() / (2 + tol);
  const int lo = nodeSize.reduce() - hi;
  int bal = accum.reduce();
  int pass = 0;
  int changed = 0;
  std::cout<<"cut  "<<calculate_cutsize(g)<<"\n";
  while (pass < 100) {
  int cut = calculate_cutsize(g);
    initGain(g, -1);
    GNodeBag bag;
    findBoundary(bag, g);
    std::vector<GNode> nodeListz;
    std::vector<GNode> nodeListo;
    std::set<GNode> nodelistz;
    std::set<GNode> nodelisto;
    unsigned zeroW = 0;
    unsigned oneW = 0;
    for (auto b : bag) {
      for (auto n : g.edges(b)) {
        auto node = g.getEdgeDst(n);
        int gainz = g.getData(node).getGain();
        if (gainz < 0) continue;
        unsigned pp = g.getData(node).getPart();
        if (pp == 0) {
          nodelistz.insert(node);
        }
        else {
          nodelisto.insert(node);
        }
      }
    }
    for (auto x : nodelisto) { 
      nodeListo.push_back(x);
      oneW += g.getData(x).getWeight();
    }
    for (auto x : nodelistz) {
      nodeListz.push_back(x);
      zeroW += g.getData(x).getWeight();
    }
    std::cout<<nodeListo.size()<<" "<<nodeListz.size()<<"\n";

    std::sort(nodeListz.begin(), nodeListz.end(), [&g] (GNode& lpw, GNode& rpw) {
      if ((float)((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) == (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()))) return (float)g.getData(lpw).nodeid < (float)g.getData(rpw).nodeid;
      return (float) ((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) > (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()));
    });
    std::sort(nodeListo.begin(), nodeListo.end(), [&g] (GNode& lpw, GNode& rpw) {
      if ((float)(g.getData(lpw).getGain() * (1.0f / g.getData(lpw).getWeight())) == (float)((g.getData(rpw).FS - g.getData(rpw).TE) * (1.0f / g.getData(rpw).getWeight())))  return (float)g.getData(lpw).nodeid < (float)g.getData(rpw).nodeid;
      return (float)((g.getData(lpw).getGain()) * (1.0f / g.getData(lpw).getWeight())) > (float)((g.getData(rpw).getGain()) * (1.0f / g.getData(rpw).getWeight()));
    });

    unsigned sizez = nodeListz.size();
    unsigned sizeo = nodeListo.size();
   // std::cout<<"size of z "<<sizez<<" "<<sizeo<<"\n";
    unsigned z = 0, o = 0;
    GNodeBag bbagz, bbago;
    int zw = 0;
    int ow = 0;
    unsigned ts = 0;
    if (zeroW <= oneW) {
      int i = 0;
      for (auto zz : nodeListz) {
        //auto zz = nodeListz.at(i);
        float gainz = (g.getData(zz).getGain()) * (1.0f / g.getData(zz).getWeight());
        if (gainz <= 0.4f) break;
        bbagz.push_back(zz);
        zw += g.getData(zz).getWeight();
        i++;
        if (i > sqrt(Size)) break;
      }
      auto ooo = nodeListo.begin();
      while (ow < zw && ooo != nodeListo.end()) {
        auto oo = *ooo;
        bbago.push_back(oo);
        ow += g.getData(oo).getWeight();
        ++ooo;
      }
      for (auto b : bbagz)  { 
        g.getData(b).setPart(1);
        g.getData(b).counter++;
      }
      for (auto b : bbago) {
        g.getData(b).setPart(0);
        g.getData(b).counter++;
        ts += g.getData(b).getWeight();
        if (ts >= zw) break;
      }
     }
     else {
        int i = 0;
        for (auto oo : nodeListo) {
           float gaino = (g.getData(oo).getGain()) * (1.0f / g.getData(oo).getWeight());
           if (gaino <= 0.4f) break;
           bbago.push_back(oo);
           ow += g.getData(oo).getWeight();
           i++;
           if (i > sqrt(Size)) break;
        }
        unsigned j = 0;
        auto ooo = nodeListz.begin();
        while (zw < ow && ooo != nodeListz.end()) {
           auto oo = *ooo;
           bbagz.push_back(oo);
           zw += g.getData(oo).getWeight();
           ooo++;
           j++;
        }
      for (auto b : bbago) { 
        g.getData(b).setPart(0);
        g.getData(b).counter++;
      }
      for (auto b : bbagz) {
        g.getData(b).setPart(1);
        g.getData(b).counter++;
        ts += g.getData(b).getWeight();
        if (ts >= zw) break;
      }
       // ow -> 1 to 0
   } 
    pass++;
    if(ts == 0) break;
  int cutb = calculate_cutsize(g);
    std::cout<<"cut size "<<calculate_cutsize(g)<<"\n";
  if (cut - cutb < 100) break;
 }
    std::cout<<changed<<"\n";
    std::cout<<"cut size "<<calculate_cutsize(g)<<"\n";
  /*galois::for_each(
    galois::iterate(nodeList), // go over cells with the highest gain first
    [&](GNode n, auto& cnx) {
      auto& nd         = g.getData(n);
      unsigned curpart = nd.getPart();
      unsigned newpart = 0;
      if (bal < hi)
        return;
      if (curpart != newpart) {
        nd.setPart(newpart);
          //bal--;
        __sync_fetch_and_sub(&bal, curpart);

      }
    //   }
    },
    galois::loopname("make balance"), galois::wl<galois::worklists::Deterministic<> >());
  }
   // more 1s 
  if (hi < bal) {
    galois::for_each(
        galois::iterate(nodeList), // go over cells with the highest gain first
        [&](GNode n, auto& cnx) {
     //  for (auto n : nodeList) {
          auto& nd         = g.getData(n);
          unsigned curpart = nd.getPart();
          unsigned newpart = 0;
          if (bal < hi)
            return;
          if (curpart != newpart) {
            nd.setPart(newpart);
            //bal--;
            __sync_fetch_and_sub(&bal, curpart);

          }
    //   }
        },
      galois::loopname("make balance"), galois::wl<galois::worklists::Deterministic<> >());
  }

  else if (bal < lo) {
    galois::for_each(
        galois::iterate(nodeList), // go over cells with the highest gain first
        [&](GNode n, auto& cnx) {
  //for (auto n : nodeList) {
          auto& nd         = g.getData(n);
          unsigned curpart = nd.getPart();
          unsigned newpart = 1;
          if (lo < bal)
            return;
          if (curpart != newpart) {
            nd.setPart(newpart);
           // bal++;
            __sync_fetch_and_add(&bal, newpart);
          }
  //}
        },
      galois::loopname("make balance"), galois::wl<galois::worklists::Deterministic<> >());
  }
  //std::cout<<"cut size is "<<calculate_cutsize(g)<<"\n";
  int pass = 0;
  int cut = 0;
  while (pass < 2) {
    GNodeBag boundary, zero, one;
    parallelBoundarySwap(boundary, g, nodeList);
    pgain(boundary, g);

    //sort the boundary in parallel
    galois::do_all(
      galois::iterate(boundary), 
      [&](GNode n) {
        unsigned part = g.getData(n).getPart();
        if (part == 0) zero.push(n);
        else one.push(n);
      },
    galois::loopname("refine"));
    std::vector<GNode> zeros;
    std::vector<GNode> ones;
    for (auto n : zero) zeros.push_back(n);
    for (auto n : one) ones.push_back(n);
    std::sort(ones.begin(), ones.end(), [&g] (GNode& lpw, GNode& rpw) {
    if (g.getData(lpw).gain == g.getData(rpw).gain) return g.getData(lpw).nodeid < g.getData(rpw).nodeid;
      return g.getData(lpw).gain > g.getData(rpw).gain;
    });


    std::sort(zeros.begin(), zeros.end(), [&g] (GNode& lpw, GNode& rpw) {
    if (g.getData(lpw).gain == g.getData(rpw).gain) return g.getData(lpw).nodeid < g.getData(rpw).nodeid;
      return g.getData(lpw).gain > g.getData(rpw).gain;
    });
    int zcount = 0;
    int ocount = 0;
    if (zeros.size() >= ones.size()) {
      galois::do_all(galois::iterate(ones),
              [&](GNode n) {
                int indx = std::distance(ones.begin(), find(ones.begin(), ones.end(), n));
                int gain = g.getData(n).gain + g.getData(zeros[indx]).gain;
                if (gain > 0) {
                  ocount++;
                  g.getData(n).setPart(0);
                  g.getData(zeros[indx]).setPart(1);
                  g.getData(n).counter++;
                  g.getData(zeros[indx]).counter++;
                } 
              },
              galois::loopname("swap"));
       }
    else {
      galois::do_all(galois::iterate(zeros),
              [&](GNode n) {
                int indx = std::distance(zeros.begin(), find(zeros.begin(), zeros.end(), n));
                int gain = g.getData(n).gain + g.getData(ones[indx]).gain;
                if (gain > 0) {
                  zcount++;
                  g.getData(n).setPart(1);
                  g.getData(ones[indx]).setPart(0);
                  g.getData(n).counter++;
                  g.getData(ones[indx]).counter++;
                } 
              },
              galois::loopname("swap"));
    }
    pass++;
  }
  */
unlock(g);
}

unsigned hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

int newRefine(GGraph& g, const float tol) {
  
    unsigned nodeSize = std::distance(g.cellList().begin(), g.cellList().end());
    //unsigned netSize = std::distance(g.getNets().begin(), g.getNets().end());
    const int hi = (1 + tol) * nodeSize / (2 + tol);
    const int lo = nodeSize - hi;
    std::map<GNode, unsigned> part;


    galois::do_all(
      galois::iterate(g.getNets()),
      [&](GNode n) {
        for (auto c : g.out_edges(n)) {
           auto cc = g.getEdgeDst(c);
           galois::atomicAdd(g.getData(cc).cnt, 1);
        }
      },
    galois::loopname("refine_cnt"));

    int prevbest, bestcut = INT_MAX;
    do {
      prevbest = bestcut;
      int bal = 0;

      galois::GAccumulator<unsigned int> accum;
      galois::do_all(galois::iterate(g.cellList()), 
            [&](GNode n) {
            g.getData(n).icnt = 1.0f / g.getData(n).cnt;
               if (g.getData(n).getPart() > 0)
               accum += 1;
            g.getData(n).sum1 = g.getData(n).getPart() * 1 ? 1 :-1;
            g.getData(n).sum2 = 0;
      },
      galois::loopname("make balance"));

      bal = accum.reduce();

      int changes;
      do {
       const float ibal0 = 1.0f / (nodeSize - bal);
       const float ibal1 = 1.0f / bal;
        const float f0 = std::min(std::max(1.0f, lo * ibal0), hi * ibal0);
        const float f1 = std::min(std::max(1.0f, lo * ibal1), hi * ibal1);
        galois::do_all(
          galois::iterate(g.getNets()),
          [&](GNode n) {
            float sum = 0;
            unsigned len = std::distance(g.edge_begin(n, galois::MethodFlag::UNPROTECTED), g.edge_end(n, galois::MethodFlag::UNPROTECTED));
            for (auto c : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
                auto cell = g.getEdgeDst(c);
                sum += g.getData(cell).sum1;
            }
            const float isizem1 = 1.0f / len;
            for (auto c : g.edges(n)) {
                auto cell = g.getEdgeDst(c);
                float off = (sum - g.getData(cell).sum1) * isizem1 * g.getData(cell).icnt;
                off *= (off < 0) ? f0 : f1;
                galois::atomicAdd(g.getData(cell).sum2, off);
            }
      },//galois::wl<galois::worklists::Deterministic<> >(),
      galois::steal(), galois::loopname("refine_off"));


      bal = 0;
      changes = 0;
      GNodeBag bag;
      galois::GAccumulator<unsigned int> accumb;
      galois::GAccumulator<unsigned int> accumc;

      galois::do_all(galois::iterate(g.cellList()), 
            [&](GNode c) {
            auto& cData = g.getData(c);
            if (cData.sum1 * cData.sum2 < 0.0f) 
              accumb += 1;//changes
            if (cData.sum2 > 0) {
              accumc += 1;//bal
              if (cData.getPart() == 0)
                bag.push(c);
            }
            else if (cData.getPart() == 1)
                bag.push(c);
            g.getData(c).sum1 = 0;
      },
      galois::loopname("changes"));

      bal = accumc.reduce();
      changes = accumb.reduce();
      for (auto c : g.cellList()) {
          std::swap(g.getData(c).sum1, g.getData(c).sum2);
          part[c] = g.getData(c).getPart();
      }
      for (auto c : bag) {
          if (g.getData(c).getPart() == 0)
          part[c] = 1;
          else part[c] = 0;
      }
        
      int count = 0;
      if ((lo <= bal) && (bal <= hi)) {
          count = calculate_cutsize(g, part);
          if (bestcut > count) {
              bestcut = count;
      galois::do_all(
          galois::iterate(g.cellList()),
         [&](GNode c) {g.getData(c).setPart(part[c]);
      },
      galois::steal(), galois::loopname("refine_setPart"));
            }
        }
      } while (changes > nodeSize / 2048);
    } while (prevbest * 0.99 > bestcut);
    return bestcut;
}

void parallel_make_balance(GGraph& g, float tol, int p) {

  unsigned Size = std::distance(g.cellList().begin(), g.cellList().end());

  galois::GAccumulator<unsigned int> accum;
  galois::GAccumulator<unsigned int> nodeSize;
  galois::do_all(galois::iterate(g.cellList()),
  [&](GNode n) {
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
	galois::do_all(galois::iterate(g.cellList()),
      [&](GNode n) {

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
    	galois::do_all(galois::iterate(g.cellList()),
      [&](GNode n) {

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

 unsigned Size = std::distance(g.cellList().begin(), g.cellList().end());

  galois::GAccumulator<unsigned int> accum;
  galois::GAccumulator<unsigned int> nodeSize;
  galois::do_all(galois::iterate(g.cellList()),
  [&](GNode n) {
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
      for (auto b : g.getNets()) {
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
      for (auto b : g.getNets()) {
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
 //   std::cout<<"inside refine\n";
    //int pass = 0 ;
   // unsigned size = 0;//std::distance(coarseGraph->getGraph()->cellList().begin(), coarseGraph->getGraph()->cellList().end());
  /*auto ggg = coarseGraph->getGraph();
  int i = 0;
    for (auto c : ggg->cellList()) {
        int sign = hash(i + 1 * size) & 1;
        ggg->getData(c).setPart(sign);
        i++;
    }*/
  //  std::cout<<"cut size is "<<calculate_cutsize(*coarseGraph->getGraph())<<"\n";
  int pass = 0;
  MetisGraph* fineG;
  do {
    fineG = coarseGraph;
    MetisGraph* fineGraph = coarseGraph->getFinerGraph();
  //unsigned newsize  = std::distance(coarseGraph->getGraph()->cellList().begin(), coarseGraph->getGraph()->cellList().end());
  auto gg = coarseGraph->getGraph();
  const float ratio = 55.0 / 45.0;  // change if needed
  const float tol = std::max(ratio, 1 - ratio) - 1;
  //galois::StatTimer T("parallel_ref");
 //T.start();
      parallel_refine_KF(*gg, tol, pass);
  //T.stop();
    //std::cout<<"PArallel reifnecut size is "<<calculate_cutsize(*coarseGraph->getGraph())<<"\n";
  //   int b = calculate_cutsize(*coarseGraph->getGraph());
   //  if (c - b < 10) break;
   // }
  //galois::StatTimer T1("balance");
  //T1.start();
 // make_balance(*gg, tol, pass);
  parallel_make_balance(*gg, tol, pass);
  //T1.stop();
    //std::cout << "refine," << T.get() << '\n';
    //std::cout << "balance:," << T1.get() << '\n';
//    std::cout<<"Balancecut size is "<<calculate_cutsize(*coarseGraph->getGraph())<<"\n";
   /*else
   // std::cout<<newRefine(*gg, tol)<<"\n";
  //else if (pass %2 == 0)
    for (int i = 0; i < 20; i++) {
     int c = calculate_cutsize(*coarseGraph->getGraph());
      refine_by_swap(*gg, tol);
     int b = calculate_cutsize(*coarseGraph->getGraph());
     if (c - b < 10) break;
    }*/
  //  refine_KF(*gg, tol);
   // pass++;
   // size = newsize;
   pass++;
    bool do_pro = true;
    if (fineGraph && do_pro) {
      projectPart(coarseGraph);
    }
   } while ((coarseGraph = coarseGraph->getFinerGraph()));
    //std::cout<<newRefine(*gg, tol)<<"\n";
/*  std::ofstream ofs("out.2.txt");
  int one = 0;
  int zero = 0;
  std::map<int, int> cell;
  GGraph* g = fineG->getGraph();
  int mys = std::distance(g->cellList().begin(), g->cellList().end());
  for (auto c : g->cellList()) {
    int id = g->getData(c).nodeid;
    int p = g->getData(c).getPart();
    if (p == 1) one++;
    else zero++;
    cell[id] = p;
  }
  for (auto mm : cell) {
    ofs<<mm.second<<"\n";  
  }
  ofs.close();*/
}

