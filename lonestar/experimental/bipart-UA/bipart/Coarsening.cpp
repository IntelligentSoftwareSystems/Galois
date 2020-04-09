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

#include "Metis.h"
#include "galois/Galois.h"
#include "galois/AtomicHelpers.h"
#include "galois/Reduction.h"
#include "galois/runtime/Profile.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/gstl.h"

#include <iostream>
#include <unordered_set>
#include <unordered_map>

int TOTALW;
int LIMIT;
bool FLAG = false;
namespace {

#ifndef NDEBUG
void assertAllMatched(GNode node, GGraph* graph) {
  for (auto jj : graph->edges(node))
    assert(node == graph->getEdgeDst(jj) ||
           graph->getData(graph->getEdgeDst(jj)).isMatched());
}

void assertNoMatched(GGraph* graph) {
  for (auto nn = graph->begin(), en = graph->end(); nn != en; ++nn)
    assert(!graph->getData(*nn).isMatched());
}
#endif


typedef galois::GAccumulator<unsigned> Pcounter;


// phaseII scheduling
void parallelReHMatchAndCreateNodes(MetisGraph* graph) {

  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  assert(fineGGraph != coarseGGraph);
  typedef std::set<GNode> SecTy;
  typedef std::vector<GNode> VecTy;
  typedef galois::substrate::PerThreadStorage<SecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  
  // Making deterministic
  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          galois::atomicMin(fineGGraph->getData(dst).netval, fineGGraph->getData(item).netval.load());
        }
      },
      galois::steal(),  galois::loopname("atomicMin2"));


  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          if (fineGGraph->getData(dst).netval == fineGGraph->getData(item).netval)
            galois::atomicMin(fineGGraph->getData(dst).netnum, fineGGraph->getData(item).netnum.load());
        }
      },
      galois::steal(),  galois::loopname("secondMin2"));
}

int hash(unsigned val) {
  unsigned long int seed = val * 1103515245 + 12345;
  return((unsigned)(seed/65536) % 32768);
}

void parallelRand(MetisGraph* graph, int iter) {

  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  int x = iter % 2;
  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
//      unsigned id = fineGGraph->getData(item).netrand;
      unsigned idx = fineGGraph->getData(item).netnum;
    //  if (idx % 2 == x)
          fineGGraph->getData(item).netrand = hash(idx);
      },
      galois::loopname("rand"));
}

using MatchingPolicy  = void(GNode, GGraph*);

void PLD_f(GNode node, GGraph* fineGGraph) {
  int ss = std::distance(fineGGraph->edge_begin(node), fineGGraph->edge_end(node));
  fineGGraph->getData(node).netval = -ss;
}
void RAND_f(GNode node, GGraph* fineGGraph) {
  unsigned id = fineGGraph->getData(node).netrand;
  fineGGraph->getData(node).netval = -id;
  fineGGraph->getData(node).netrand = -fineGGraph->getData(node).netnum; 
}
void PP_f(GNode node, GGraph* fineGGraph) {
  int ss = std::distance(fineGGraph->edge_begin(node), fineGGraph->edge_end(node));
  fineGGraph->getData(node).netval = ss;
}
void WD_f(GNode node, GGraph* fineGGraph) {
  int w = 0;
  for (auto n : fineGGraph->edges(node)) {
    auto nn = fineGGraph->getEdgeDst(n);
    w += fineGGraph->getData(nn).getWeight();
  }
  fineGGraph->getData(node).netval = -w;
}
void MWD_f(GNode node, GGraph* fineGGraph) {
  int w = 0;
  for (auto n : fineGGraph->edges(node)) {
    auto nn = fineGGraph->getEdgeDst(n);
    w += fineGGraph->getData(nn).getWeight();
  }
  fineGGraph->getData(node).netval = w;
}
void RI_f(GNode node, GGraph* fineGGraph) {
  int ss = std::distance(fineGGraph->edge_begin(node), fineGGraph->edge_end(node));
  fineGGraph->getData(node).netval = ss;
}
void MRI_f(GNode node, GGraph* fineGGraph) {
  int ss = std::distance(fineGGraph->edge_begin(node), fineGGraph->edge_end(node));
  fineGGraph->getData(node).netval = ss;
}
void DEG_f(GNode node, GGraph* fineGGraph) {
  int w = 0;
  int ss = std::distance(fineGGraph->edge_begin(node), fineGGraph->edge_end(node));
  fineGGraph->getData(node).netval = ss;
  for (auto n : fineGGraph->edges(node)) {
    auto nn = fineGGraph->getEdgeDst(n);
    w += fineGGraph->getData(nn).getWeight();
  }
  fineGGraph->getData(node).netval = - (w / ss);
}
void MDEG_f(GNode node, GGraph* fineGGraph) {
  int w = 0;
  int ss = std::distance(fineGGraph->edge_begin(node), fineGGraph->edge_end(node));
  fineGGraph->getData(node).netval = ss;
  for (auto n : fineGGraph->edges(node)) {
    auto nn = fineGGraph->getEdgeDst(n);
    w += fineGGraph->getData(nn).getWeight();
  }
  fineGGraph->getData(node).netval = w/ss;
}



template <MatchingPolicy matcher> 
void parallelPrioRand(MetisGraph* graph, int iter) {

  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  assert(fineGGraph != coarseGGraph);
  typedef std::set<GNode> SecTy;
  typedef std::vector<GNode> VecTy;
  typedef galois::substrate::PerThreadStorage<SecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  std::string name = "phaseI";
  parallelRand(graph, iter); 
  
  // Making deterministic
  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
          matcher(item, fineGGraph);
          for (auto c : fineGGraph->edges(item)) {
            auto dst = fineGGraph->getEdgeDst(c);
            galois::atomicMin(fineGGraph->getData(dst).netval, fineGGraph->getData(item).netval.load());
          }
      },
      galois::steal(),  galois::loopname("atomicMin"));

  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
            for (auto c : fineGGraph->edges(item)) {
                auto dst = fineGGraph->getEdgeDst(c);
                if (fineGGraph->getData(dst).netval == fineGGraph->getData(item).netval)
                galois::atomicMin(fineGGraph->getData(dst).netrand, fineGGraph->getData(item).netrand.load());
            }  
     },
      galois::steal(),  galois::loopname("secondMin2"));
  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          if (fineGGraph->getData(dst).netrand == fineGGraph->getData(item).netrand)
            galois::atomicMin(fineGGraph->getData(dst).netnum, fineGGraph->getData(item).netnum.load());
        }
      },
      galois::steal(),  galois::loopname("secondMin"));
}


// hyper edge matching
template <MatchingPolicy matcher> 
void parallelHMatchAndCreateNodes(MetisGraph* graph,
                                 int iter) {
  parallelPrioRand<matcher>(graph, iter);

  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  assert(fineGGraph != coarseGGraph);
  typedef std::set<GNode> SecTy;
  typedef std::vector<GNode> VecTy;
  typedef galois::substrate::PerThreadStorage<SecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  std::string name = "phaseI";
  // hyperedge coarsening 
  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
         
        unsigned id = fineGGraph->getData(item).netnum;
       // if (fmod(id, pow(2,iter)) <= pow(2,(iter - 1)) - 1) { //final
           // leave the hedges
    /*        GNode netN;
            MetisNode n2;
            n2.netval = INT_MAX;
            n2.netrand = fineGGraph->getData(item).netrand;
            n2.netnum = fineGGraph->getData(item).netnum;
            n2.setChild(item);
            netN = coarseGGraph->createNode(n2);
            coarseGGraph->addNode(netN);
            coarseGGraph->addHyperedge(netN);
            fineGGraph->getData(item).setParent(netN);
            fineGGraph->getData(item).setMatched();
      *///    return;
       // }
        bool flag = false;
        auto& edges = *edgesThreadLocal.getLocal();
        edges.clear();
        int w = 0;
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          auto& data = fineGGraph->getData(dst);
          if (data.isMatched()) { flag = true;
            continue;//return;
          }
          if (data.netnum == fineGGraph->getData(item).netnum) {
      //      if (w + fineGGraph->getData(dst).getWeight() > LIMIT) break;
            edges.insert(dst);
            w += fineGGraph->getData(dst).getWeight();
          }
          else {
              //edges.clear();
              //break;
              flag = true;
          }
        }

        if (!edges.empty()) {
          if (flag && edges.size() == 1) return; 
          fineGGraph->getData(item).setMatched();
          GNode N;
          MetisNode n1;
          n1.netval = INT_MAX;
          n1.netnum = INT_MAX;
          n1.nodeid = INT_MAX;
          n1.netrand = INT_MAX;
          N = coarseGGraph->createNode(n1);
          coarseGGraph->addNode(N);
          coarseGGraph->addCell(N);
          int ww = 0;
          // create cell node
          for (auto pp : edges) {
            coarseGGraph->getData(N).nodeid = std::min(coarseGGraph->getData(N).nodeid, fineGGraph->getData(pp).nodeid);
            ww += fineGGraph->getData(pp).getWeight();
            fineGGraph->getData(pp).setMatched();
            fineGGraph->getData(pp).setParent(N);
            fineGGraph->getData(pp).netnum = fineGGraph->getData(item).netnum;
          }
          coarseGGraph->getData(N).setWeight(ww);
          if (flag) {
            GNode netN;
            MetisNode n2;
            n2.netval = INT_MAX;
            n2.netrand = fineGGraph->getData(item).netrand;
            n2.netnum = fineGGraph->getData(item).netnum;
            n2.setChild(item);
            netN = coarseGGraph->createNode(n2);
            coarseGGraph->addNode(netN);
            coarseGGraph->addHyperedge(netN);
            fineGGraph->getData(item).setParent(netN);
            //fineGGraph->getData(item).setMatched();
          }
        }
      },
      galois::loopname("phaseI"));

}

void moreCoarse(MetisGraph* graph, int iter) {
  
  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  assert(fineGGraph != coarseGGraph);
  typedef std::set<int> SecTy;
  typedef std::vector<GNode> VecTy;
  GNodeBag bag;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
        if (fineGGraph->getData(item).isMatched()) return;
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          if (fineGGraph->getData(dst).isMatched()) 
              fineGGraph->getData(dst).netval = INT_MIN;
        }
      },
      galois::loopname("atomicMin2"));

  galois::do_all( 
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
          if (fineGGraph->getData(item).isMatched()) return;
          auto& cells = *edgesThreadLocal.getLocal();
          cells.clear();
          int best = INT_MAX;
          GNode b;
          int w = 0;
          for (auto edge : fineGGraph->edges(item)) {
	      auto e = fineGGraph->getEdgeDst(edge);
              auto& data = fineGGraph->getData(e);
              if (!fineGGraph->getData(e).isMatched()) {
                  if (data.netnum == fineGGraph->getData(item).netnum) {
                      cells.push_back(e);
                      w += coarseGGraph->getData(e).getWeight();
                  }
              }
              else if (fineGGraph->getData(e).netval == INT_MIN) {
                  auto nn = fineGGraph->getData(e).getParent();
       //           if (coarseGGraph->getData(nn).getWeight() + w > LIMIT) continue;
                  if (fineGGraph->getData(e).getWeight() < best) {
                    best = fineGGraph->getData(e).getWeight();
                    b = e;
                  }
                  else if (fineGGraph->getData(e).getWeight() == best) {
                    if (fineGGraph->getData(e).nodeid < fineGGraph->getData(b).nodeid)
                   // if(e < b)
											b = e;
                  }
              }

          }
          if (cells.size() > 0) {
              if (best < INT_MAX) {
                  auto nn = fineGGraph->getData(b).getParent();
                  int ww = coarseGGraph->getData(nn).getWeight();
                  for (auto e : cells) {
	            bag.push(e);
                    fineGGraph->getData(e).setMatched();
                    fineGGraph->getData(e).setParent(nn);
                    fineGGraph->getData(e).netnum = fineGGraph->getData(b).netnum;
                  }
                             
              }        
          }          
      },
        galois::loopname("moreCoarse"));
      for (auto c : bag) {
        auto nn = fineGGraph->getData(c).getParent();
      //  coarseGGraph->getData(nn).nodeid = std::min(coarseGGraph->getData(nn).nodeid, fineGGraph->getData(c).nodeid);
        int ww = coarseGGraph->getData(nn).getWeight();
        ww += fineGGraph->getData(c).getWeight();
        coarseGGraph->getData(nn).setWeight(ww);
      }
}

// Coarsening phaseII
void coarsePhaseII(MetisGraph* graph,
                    int iter) {

  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  assert(fineGGraph != coarseGGraph);
  typedef std::set<int> SecTy;
  typedef std::vector<GNode> VecTy;
  typedef galois::substrate::PerThreadStorage<SecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalDataV;
  ThreadLocalDataV edgesThreadLocalV;
  std::string name = "CoarseningPhaseII";
  moreCoarse(graph, iter);

  galois::do_all( 
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
        if (fineGGraph->getData(item).isMatched()) return;
        unsigned id = fineGGraph->getData(item).netnum;
        unsigned ids;
        int count = 0;
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          auto& data = fineGGraph->getData(dst);
          if (data.isMatched()) {
            if (count == 0) {
              ids = coarseGGraph->getData(data.getParent()).nodeid;
              count++;
            }
            else if (ids != coarseGGraph->getData(data.getParent()).nodeid) {
              count++;
              break;
            }
          }
          else { 
              count = 0;
              break;
          }
        }
        if (count == 1) {
            fineGGraph->getData(item).setMatched();
        }
        else {
          GNode netN;
          MetisNode n2;
          n2.netval = INT_MAX;
          n2.netnum = fineGGraph->getData(item).netnum;
          n2.netrand = fineGGraph->getData(item).netrand;
          n2.setChild(item);
          netN = coarseGGraph->createNode(n2);
          coarseGGraph->addNode(netN);
          coarseGGraph->addHyperedge(netN);
          fineGGraph->getData(item).setParent(netN);
          fineGGraph->getData(item).setMatched();
        }

      },
      galois::loopname("phaseII_2"));

	galois::GAccumulator<unsigned> h;
  galois::do_all(
      galois::iterate(fineGGraph->cellList()),
      [&](GNode ii) {
          if (!fineGGraph->getData(ii).isMatched()) {
            MetisNode n1;
            n1.netval = INT_MAX;
            n1.netnum = INT_MAX;
            n1.nodeid = fineGGraph->getData(ii).nodeid;
            n1.netrand = INT_MAX;
            unsigned val = fineGGraph->getData(ii).getWeight();
            n1.setWeight(val);
            GNode N = coarseGGraph->createNode(n1);
            coarseGGraph->addNode(N);
            coarseGGraph->addCell(N);
            fineGGraph->getData(ii).setMatched();
            fineGGraph->getData(ii).setParent(N);
						h+=1;
          }
      },
      galois::loopname("noedgebag match"));

		std::cout <<"h: " << h.reduce() << std::endl;
}

void parallelCreateEdges(MetisGraph* graph) {

  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  assert(fineGGraph != coarseGGraph);
    galois::do_all(
      galois::iterate(coarseGGraph->getNets()),
      [&](GNode item) {
        MetisNode& nodeData =
            coarseGGraph->getData(item, galois::MethodFlag::UNPROTECTED);
        for (auto c : fineGGraph->edges(nodeData.getChild(0))) {
          auto dst = fineGGraph->getEdgeDst(c);
          auto& data = fineGGraph->getData(dst);
          auto cpar = data.getParent();
            coarseGGraph->addEdge(item, cpar);
        } 
      },
      galois::loopname("HyperedgeEDGE"));
}


void findMatching(MetisGraph* coarseMetisGraph,
                       scheduleMode sch,
                       int iter) {
  MetisGraph* fineMetisGraph = coarseMetisGraph->getFinerGraph();

  GNodeBag bagOfLoners;
  Pcounter pc;
  unsigned c;

  bool useOBIM = true;

      switch(sch) {
           case PLD:
            parallelHMatchAndCreateNodes<PLD_f>(coarseMetisGraph,
                                            iter);
       break;
           case RAND:
            parallelHMatchAndCreateNodes<RAND_f>(coarseMetisGraph,
                                            iter);
       break;
           case PP:
             parallelHMatchAndCreateNodes<PP_f>(coarseMetisGraph,
                                            iter);
       break;
           case WD:
             parallelHMatchAndCreateNodes<WD_f>(coarseMetisGraph,
                                            iter);
       break;
           case RI:
             parallelHMatchAndCreateNodes<RI_f>(coarseMetisGraph,
                                            iter);
       break;
           case MRI:
             parallelHMatchAndCreateNodes<MRI_f>(coarseMetisGraph,
                                            iter);
       break;
           case MWD:
             parallelHMatchAndCreateNodes<MWD_f>(coarseMetisGraph,
                                            iter);
       break;
           case DEG:
             parallelHMatchAndCreateNodes<DEG_f>(coarseMetisGraph,
                                            iter);
       break;
           case MDEG:
             parallelHMatchAndCreateNodes<MDEG_f>(coarseMetisGraph,
                                            iter);
       break;
      
       }
       coarsePhaseII(coarseMetisGraph, iter);
       parallelCreateEdges(coarseMetisGraph);
}

MetisGraph* coarsenOnce(MetisGraph* fineMetisGraph, scheduleMode sch, 
                         int iter) {
  MetisGraph* coarseMetisGraph = new MetisGraph(fineMetisGraph);
  findMatching(coarseMetisGraph,sch, iter);
  return coarseMetisGraph;
}

} // namespace

MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo,
                    scheduleMode sch) {

  MetisGraph* coarseGraph = fineMetisGraph;
  unsigned size           = std::distance(fineMetisGraph->getGraph()->cellList().begin(), fineMetisGraph->getGraph()->cellList().end());
  const float ratio = 55.0 / 45.0;  // change if needed
  const float tol = std::max(ratio, 1 - ratio) - 1;
  const int hi = (1 + tol) * size / (2 + tol);
  const int lo = size - hi;
  LIMIT = hi / 4;
  int totw = 0;
  
  unsigned Size = size;
  unsigned iterNum        = 0;
  unsigned newSize = size;
  while (size > coarsenTo) { 
    if (iterNum > coarsenTo) break;
    if (Size - newSize <= 0 && iterNum > 2) break; //final
     Size = newSize;
     coarseGraph      = coarsenOnce(coarseGraph, sch, iterNum);
      newSize = std::distance(coarseGraph->getGraph()->cellList().begin(), coarseGraph->getGraph()->cellList().end());
      if (newSize < coarsenTo)break;
      int netsize           = std::distance(coarseGraph->getGraph()->getNets().begin(), coarseGraph->getGraph()->getNets().end());
     std::cout<<"SIZE IS "<<newSize<<"and net is "<<netsize<<"\n";
    ++iterNum;
    
  }
  return coarseGraph;
}
