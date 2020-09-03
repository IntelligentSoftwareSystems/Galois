#include "Metis.h"
#include "galois/Galois.h"
#include "galois/AtomicHelpers.h"
#include "galois/Reduction.h"
#include "galois/runtime/Profile.h"
#include "galois/substrate/PerThreadStorage.h"
#include "galois/gstl.h"

#include <iostream>
#include <float.h>
#include <unordered_set>
#include <unordered_map>

namespace {

const int dim = 200;
double DB_MAX;
int computeScore(GGraph& g, GNode n, GNode m) {
  return (abs(g.getData(n).emb - g.getData(m).emb));
}



void phaseI(MetisGraph* graph, int iter) {
  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  assert(fineGGraph != coarseGGraph);
  GNodeBag bag;
  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&] (GNode item) {
        double avg = 0.0f;
        unsigned size = 0;
        for (auto n : fineGGraph->edges(item)) {
          auto node = fineGGraph->getEdgeDst(n);
          avg += fineGGraph->getData(node).emb;
          size++;
        }
        avg = avg / size;
        fineGGraph->getData(item).emb = avg;
        fineGGraph->getData(item).netval = avg;
      },
  galois::steal(), galois::loopname("match"));

  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          double score = computeScore(*fineGGraph, item, dst);
          galois::atomicMin(fineGGraph->getData(dst).netval, score);
        }
      },
  galois::steal(),  galois::loopname("atomicMin"));
  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
        for (auto c : fineGGraph->edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          if (fineGGraph->getData(dst).netval == fineGGraph->getData(item).netval)
            galois::atomicMin(fineGGraph->getData(dst).netnum, fineGGraph->getData(item).netnum.load());
        }
      },
      galois::steal(),  galois::loopname("secondMin"));
  typedef std::vector<GNode> VecTy;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalData;
  ThreadLocalData edgesThreadLocal;
  galois::do_all(
      galois::iterate(fineGGraph->getNets()),
      [&](GNode item) {
        unsigned id = fineGGraph->getData(item).netnum;
        if (fmod(id, pow(2,iter)) <= pow(2,(iter - 1)) - 1) { //final
           // leave the hedges
            GNode netN;
            MetisNode n2;
            n2.netval = INT_MAX;
            n2.netnum = fineGGraph->getData(item).netnum;
            n2.setChild(item);
            netN = coarseGGraph->createNode(n2);
            coarseGGraph->addNode(netN);
            coarseGGraph->addHyperedge(netN);
            coarseGGraph->getData(netN).emb = fineGGraph->getData(item).emb;
            fineGGraph->getData(item).setParent(netN);
            fineGGraph->getData(item).setMatched();
          return;
        }
         
        bool flag = false;
        auto& edges = *edgesThreadLocal.getLocal();
        edges.clear();
        int w = 0;
        for (auto c : fineGGraph->out_edges(item)) {
          auto dst = fineGGraph->getEdgeDst(c);
          auto& data = fineGGraph->getData(dst);
          if (data.isMatched()) { flag = true;
            continue;
          }
          if (data.netnum == fineGGraph->getData(item).netnum) {
            edges.push_back(dst);
            w += fineGGraph->getData(dst).getWeight();
          }
          else {
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
          N = coarseGGraph->createNode(n1);
          coarseGGraph->addNode(N);
          coarseGGraph->addCell(N);
          int ww = 0;
          int emb = 0;
          // create cell node
          for (auto pp : edges) {
            coarseGGraph->getData(N).nodeid = std::min(coarseGGraph->getData(N).nodeid, fineGGraph->getData(pp).nodeid);
            ww += fineGGraph->getData(pp).getWeight();
            fineGGraph->getData(pp).setMatched();
            fineGGraph->getData(pp).setParent(N);
            fineGGraph->getData(pp).netnum = fineGGraph->getData(item).netnum;
            emb += fineGGraph->getData(pp).emb;
          }
          coarseGGraph->getData(N).setWeight(ww);
          coarseGGraph->getData(N).emb = emb / edges.size();
          if (flag) {
            GNode netN;
            MetisNode n2;
            n2.netval = INT_MAX;
            n2.netnum = fineGGraph->getData(item).netnum;
            n2.setChild(item);
            netN = coarseGGraph->createNode(n2);
            coarseGGraph->addNode(netN);
            coarseGGraph->addHyperedge(netN);
            fineGGraph->getData(item).setParent(netN);
          }
        }
      },
      galois::steal(),  galois::loopname("phaseI"));
}

void moreCoarse(MetisGraph* graph) {
  
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
      galois::steal(),  galois::loopname("atomicMin2"));

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
                  if (fineGGraph->getData(e).getWeight() < best) {
                    best = fineGGraph->getData(e).getWeight();
                    b = e;
                  }
                  else if (fineGGraph->getData(e).getWeight() == best) {
                    if (fineGGraph->getData(e).nodeid < fineGGraph->getData(b).nodeid)
                      b = e;
                  }
              }

          }
          if (cells.size() > 0) {
              if (best < INT_MAX) {
                  int emb = 0;
                  auto nn = fineGGraph->getData(b).getParent();
                  int ww = coarseGGraph->getData(nn).getWeight();
                  for (auto e : cells) {
	            bag.push(e);
                    fineGGraph->getData(e).setMatched();
                    fineGGraph->getData(e).setParent(nn);
                    emb += fineGGraph->getData(e).emb;
                    fineGGraph->getData(e).netnum = fineGGraph->getData(b).netnum;
                  }
                  coarseGGraph->getData(nn).emb = 
                       (coarseGGraph->getData(nn).emb + emb) / cells.size();
                             
              }        
          }          
      },
        galois::steal(), galois::loopname("moreCoarse"));
      for (auto c : bag) {
        auto nn = fineGGraph->getData(c).getParent();
        coarseGGraph->getData(nn).nodeid = std::min(coarseGGraph->getData(nn).nodeid, fineGGraph->getData(c).nodeid);
        int ww = coarseGGraph->getData(nn).getWeight();
        ww += fineGGraph->getData(c).getWeight();
        coarseGGraph->getData(nn).setWeight(ww);
      }
}
void phaseII(MetisGraph* graph) {

  GGraph* fineGGraph   = graph->getFinerGraph()->getGraph();
  GGraph* coarseGGraph = graph->getGraph();
  assert(fineGGraph != coarseGGraph);
  typedef std::set<int> SecTy;
  typedef std::vector<GNode> VecTy;
  typedef galois::substrate::PerThreadStorage<VecTy> ThreadLocalDataV;
  ThreadLocalDataV edgesThreadLocalV;
  galois::GAccumulator<int> nedges;
  galois::GAccumulator<int> zero;
  galois::GAccumulator<int> colnodes;
  galois::GAccumulator<int> newn;
  std::string name = "CoarseningPhaseII";
  GNodeBag bag,nnbg,lonebag,nnbag,tbag;
  moreCoarse(graph);
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
            n2.setChild(item);
            n2.emb = fineGGraph->getData(item).emb;
            netN = coarseGGraph->createNode(n2);
            coarseGGraph->addNode(netN);
            coarseGGraph->addHyperedge(netN);
            fineGGraph->getData(item).setParent(netN);
            fineGGraph->getData(item).setMatched();
        }
      },
      galois::steal(), galois::loopname("phaseII_2"));

  galois::do_all(
      galois::iterate(fineGGraph->cellList()),
[&](GNode ii) {
        if (!fineGGraph->getData(ii).isMatched()) {
          MetisNode n1;
          n1.netval = INT_MAX;
          n1.netnum = INT_MAX;
          n1.nodeid = fineGGraph->getData(ii).nodeid;
          n1.emb = fineGGraph->getData(ii).emb;
          unsigned val = fineGGraph->getData(ii).getWeight();
          n1.setWeight(val);
          GNode N = coarseGGraph->createNode(n1);
          coarseGGraph->addNode(N);
          coarseGGraph->addCell(N);
          fineGGraph->getData(ii).setMatched();
          fineGGraph->getData(ii).setParent(N);
        }
      },
      galois::steal(), galois::loopname("noedgebag match"));
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



void findMatching(MetisGraph* coarseMetisGraph, bool useRM, bool use2Hop,
                      int verbose, scheduleMode sch,
                      scheduleModeII phsch, int iter, int graphSize) {


  GNodeBag bagOfLoners;
  MetisGraph* fineMetisGraph = coarseMetisGraph->getFinerGraph();
  phaseI(coarseMetisGraph, iter);
  phaseII(coarseMetisGraph);
  parallelCreateEdges(coarseMetisGraph);

}

MetisGraph* coarsenOnce(MetisGraph* fineMetisGraph, unsigned& rem, bool useRM,
                        bool with2Hop, int verbose,  scheduleMode sch, 
                        scheduleModeII phsch, int iter, int graphSize) {
  MetisGraph* coarseMetisGraph = new MetisGraph(fineMetisGraph);
  findMatching(coarseMetisGraph, useRM, with2Hop, verbose, sch, phsch, iter, graphSize);
  return coarseMetisGraph;
}

} // namespace

MetisGraph* coarsen(MetisGraph* fineMetisGraph, unsigned coarsenTo,
                    bool v, scheduleMode sch, scheduleModeII phsch) {

  MetisGraph* coarseGraph = fineMetisGraph;
  unsigned size           = std::distance(fineMetisGraph->getGraph()->cellList().begin(), fineMetisGraph->getGraph()->cellList().end());
  unsigned Size = size;
  unsigned tot_size           = size;
  unsigned iterNum        = 0;
  bool with2Hop           = false;
  unsigned stat           = 0;
  bool test = false;
  unsigned newSize = size;
  bool flag = false;
  int verbose = 0;
  auto max = *coarseGraph->getGraph()->cellList().begin();
  DB_MAX = coarseGraph->getGraph()->getData(max).netval;
while (iterNum < 25) { 
    if (Size == newSize && iterNum > 2) break;
    std::map<int, int> maps;
     unsigned rem     = 0;
     Size = newSize;
      coarseGraph      = coarsenOnce(coarseGraph, rem, false, with2Hop, verbose, sch, phsch, iterNum, size);
      newSize = std::distance(coarseGraph->getGraph()->cellList().begin(), coarseGraph->getGraph()->cellList().end());
     std::cout<<"SIZE IS "<<newSize<<" \n";
    ++iterNum;
    
  }
  return coarseGraph;
}
