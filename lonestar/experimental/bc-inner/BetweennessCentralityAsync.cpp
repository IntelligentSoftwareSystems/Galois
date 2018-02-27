/** Async Betweenness centrality Node  -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
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
 *
 * Node for asynchrounous betweeness-centrality. 
 *
 * @author Dimitrios Prountzos <dprountz@cs.utexas.edu> (Main code writer)
 * @author Loc Hoang <l_hoang@utexas.edu>
 */

#define NDEBUG // Used in Debug build to prevent things from printing

#include "Lonestar/BoilerPlate.h"
#include "galois/ConditionalReduction.h"

#include <boost/tuple/tuple.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include "control.h"
#include "ND.h"
#include "ED.h"
#include "BCGraph.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////
// Command line parameters
////////////////////////////////////////////////////////////////////////////////

namespace cll = llvm::cl;

static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input graph in Galois bin "
                                                "format>"),
                                      cll::Required);

static cll::opt<unsigned int> startNode("startNode", 
                                        cll::desc("Node to start search from"),
                                        cll::init(0));

// TODO better description
static cll::opt<unsigned> buckets("buckets",
                                  cll::desc("Number of buckets to use"),
                                  cll::init(1));
static cll::opt<bool> generateCert("generateCertificate",
                                   cll::desc("Prints certificate at end of "
                                             "execution"),
                                   cll::init(false));
// TODO better description
static cll::opt<bool> useNodeBased("useNodeBased",
                                   cll::desc("Use node based execution"),
                                   cll::init(true));

#define USE_NODE_BASED
//#define INLINE_US

// TODO move BC stuff into a struct instead of having them be globals
int DEF_DISTANCE;

BCGraph* graph;
ND* currSrcNode;

// TODO define in a struct so that they can actually be initialized
using Counter = 
  ConditionalAccumulator<galois::GAccumulator<unsigned long>, COUNT_ACTIONS>;
Counter action1cnt;
Counter action2cnt;
Counter action2Excnt;
Counter action3cnt;
Counter action4cnt;
Counter action555;
Counter actionNone;

using MaxCounter = 
  ConditionalAccumulator<galois::GReduceMax<unsigned long>, COUNT_ACTIONS>;
MaxCounter largestNodeDist;

using FringeCounter = 
  ConditionalAccumulator<galois::GAccumulator<unsigned long>, DBG_FRINGECNT>;
FringeCounter fringeCnts;


struct f2Item {
  bool isCleanup;
  ND *node;
  double d;
  //f2Item(bool is, ND *nd, double _d) : isCleanup(is), node(nd), d(_d) {}
};

struct firstForEachNodeBased {
  template<typename Context>
  void operator()(ND* srcD, Context& ctx) {
    int idx = srcD->id;
    #if USE_MARKING
    srcD->markOut();
    #endif
    //assert(idx <= nnodes-1);

    int* outIdx = graph->outIdx;
    int startE = outIdx[idx];
    int endE = outIdx[idx + 1];
    ED* edgeData = graph->edgeData;

    // loop through all edges
    for (int i = startE; i < endE; i++) {
      ED& ed = edgeData[i];
      ND* dstD = ed.dst;
      
      if (srcD == dstD) continue; // ignore self loops

      // lock in set order to prevent deadlock (lower id first)
      if (CONCURRENT) {
        ND* loser;
        ND* winner;

        if (srcD < dstD) { 
          loser = srcD; winner = dstD;
        } else { 
          loser = dstD; winner = srcD;
        }

        loser->lock();
        winner->lock();
      }

      const int elevel = ed.level;
      ND *A = srcD;
      const int ADist = srcD->distance;
      ND *B = dstD;
      const int BDist = dstD->distance;

      if (BDist - ADist > 1) { // Rule 1 + Rule 3 combined
        action1cnt.update(1);

        galois::gDebug("Rule 1+3 (", A->toString(), " ||| ", B->toString(), 
                       ") ", elevel);

        A->nsuccs++;
        const double ASigma = A->sigma;
        if (CONCURRENT) { A->unlock(); }
        ND::predTY & Bpreds = B->preds;
        bool bpredsNotEmpty = !Bpreds.empty();
        //if (bpredsNotEmpty) galois::gDebug(Bpreds.size());
        Bpreds.clear();
        Bpreds.push_back(A);
        B->distance = ADist + 1;
        //int newBDist = ADist + 1;
        B->nsuccs = 0;
        B->sigma = ASigma;
        ed.val = ASigma;
        ed.level = ADist;
        if (CONCURRENT) { B->unlock(); }
        #if USE_MARKING          
        if (!B->isAlreadyIn()) 
        #endif
          ctx.push(B);
        if (bpredsNotEmpty) {
          int idx = B->id;
          int *inIdx = graph->inIdx;
          int *ins = graph->ins; 
          int startInE = inIdx[idx];
          int endInE = inIdx[idx + 1];
          for (int j = startInE; j < endInE; j++) {
            ED & inE = edgeData[ins[j]];
            ND *inNbr = inE.src;
            if (inNbr == B) continue;
            ND * aL;
            ND *aW;
            if (B < inNbr) { aL = B; aW = inNbr;} 
            else { aL = inNbr; aW = B; }
            aL->lock(); aW->lock();
            const int elev = inE.level; 
            action4cnt.update(1);
            if (inNbr->distance >= B->distance) { // Rule 4
              if (CONCURRENT) { B->unlock(); }

              galois::gDebug("Rule 4 (", inNbr->toString(), " ||| ", 
                             B->toString(), ")", elevel);

              if (elev != DEF_DISTANCE) {
                inE.level = DEF_DISTANCE;
                if (elev == inNbr->distance) {
                  action555.update(1);
                  inNbr->nsuccs--;
                }
              }
              if (CONCURRENT) { inNbr->unlock(); }
            } else { aL->unlock(); aW->unlock(); }
          }
        }
      } else if (elevel == ADist && BDist == ADist + 1) { // Rule 2: BDist = ADist + 1 and elevel = ADist
        action2cnt.update(1);
        galois::gDebug("Rule 2 (", A->toString(), " ||| ", B->toString(), ") ",
                       elevel);
        const double ASigma = A->sigma;
        const double eval = ed.val;
        const double diff = ASigma - eval;
        bool BSigmaChanged = diff >= 0.00001;
        if (CONCURRENT) { A->unlock(); }
        if (BSigmaChanged) {
        action2Excnt.update(1);
          ed.val = ASigma;
          B->sigma += diff;
          int nbsuccs = B->nsuccs;
          if (CONCURRENT) { B->unlock(); }
          if (nbsuccs > 0) {
              #ifdef INLINE_US
              int idx = B->id;
              int startE = outIdx[idx];
              int endE = outIdx[idx + 1];
              ED * edgeData = graph->edgeData;
              for (int i = startE; i < endE; i++) {
                ED & ed = edgeData[i];
                ND * dstD = ed.dst;

                if (B == dstD) continue;

                if (CONCURRENT) {
                  ND *loser, *winner;
                  if (B < dstD) { loser = B; winner = dstD;} 
                  else { loser = dstD; winner = B; }
                  loser->lock();
                  winner->lock();
                }
                const int srcdist = B->distance;
                const int dstdist = dstD->distance;
                const int elevel = ed.level;
                const int BDist = srcdist;
                ND * C = dstD; const int CDist = dstdist;
                if (elevel == BDist && CDist == BDist + 1) { // Rule 2: BDist = ADist + 1 and elevel = ADist
                  galois::gDebug("Rule 2 (", A->toString(), " ||| ", 
                                 B->toString(), ") ", elevel);
                  const double BSigma = B->sigma;
                  const double eval = ed.val;
                  const double diff = BSigma - eval;
                  if (CONCURRENT) { B->unlock(); }
                  ed.val = BSigma;
                  C->sigma += diff;
                  int ncsuccs = C->nsuccs;
                  if (CONCURRENT) { C->unlock(); }
                  if (ncsuccs > 0)
                    #if USE_MARKING
                    if (!C->isAlreadyIn()) 
                    #endif
                      ctx.push(C);
                } else {
                  B->unlock(); C->unlock();
                }
              }
#else         
#if USE_MARKING
              if (!B->isAlreadyIn()) 
#endif
                ctx.push(B);
#endif
            }
          } else {
            if (CONCURRENT) { B->unlock(); }
          }
        } else if (BDist == ADist + 1 && elevel != ADist) {  // Rule 3 // BDist = ADist + 1 and elevel != ADist
          action3cnt.update(1);
          A->nsuccs++;
          const double ASigma = A->sigma;
          if (CONCURRENT) { A->unlock(); }

          //if (AnotPredOfB) {
          B->preds.push_back(A);
          //}
          const double BSigma = B->sigma;
          galois::gDebug("Rule 3 (", A->toString(), " ||| ", B->toString(), 
                         ") ", elevel);
          B->sigma = BSigma + ASigma;
          ed.val = ASigma;
          ed.level = ADist;
          int nbsuccs = B->nsuccs;
          if (CONCURRENT) { B->unlock(); }
          //const bool BSigmaChanged = ASigma >= 0.00001;
          if (nbsuccs > 0 /*&& BSigmaChanged*/) {
#ifdef INLINE_US
              int idx = B->id;
              int startE = outIdx[idx];
              int endE = outIdx[idx + 1];
              ED * edgeData = graph->edgeData;
              for (int i = startE; i < endE; i++) {
                ED & ed = edgeData[i];
                ND * dstD = ed.dst;

                if (B == dstD) continue;

                if (CONCURRENT) {
                  ND *loser, *winner;
                  if (B < dstD) { loser = B; winner = dstD;} 
                  else { loser = dstD; winner = B; }
                  loser->lock();
                  winner->lock();
                }
                const int srcdist = B->distance;
                const int dstdist = dstD->distance;
                const int elevel = ed.level;
                const int BDist = srcdist;
                ND * C = dstD; const int CDist = dstdist;
                if (elevel == BDist && CDist == BDist + 1) { // Rule 2: BDist = ADist + 1 and elevel = ADist
                  galois::gDebug("Rule 2 (", A->toString(), " ||| ", 
                                 B->toString(), ") ", elevel);
                  const double BSigma = B->sigma;
                  const double eval = ed.val;
                  const double diff = BSigma - eval;
                  if (CONCURRENT) { B->unlock(); }
                  ed.val = BSigma;
                  C->sigma += diff;
                  int ncsuccs = C->nsuccs;
                  if (CONCURRENT) { C->unlock(); }
                  if (ncsuccs > 0)
#if USE_MARKING
                    if (!C->isAlreadyIn()) 
#endif
                      ctx.push(C);
                } else {
                  B->unlock(); C->unlock();
                }
              }
#else         
#if USE_MARKING
            if (!B->isAlreadyIn()) 
#endif
              ctx.push(B);
#endif
          }
        } else {
          actionNone.update(1);
          A->unlock(); B->unlock();
        }
      }
  }    
};

struct secondForEach {
  template<typename Context>
    void inline /*__attribute__((noinline))*/ operator()(ND *A, Context& ctx) {
      if (CONCURRENT) { A->lock(); }
      if (A->nsuccs == 0 /*&& !A->deltaDone()*/) {
        //...A->nsuccs = -1;
        //A->setDeltaDoneT();//A.deltaDone = true;
        galois::gDebug("RULE 1 ", A->toString());
        const double Adelta = A->delta;
//        if (A != currSrcNode) {
          A->bc += Adelta;
//        }
        //const double ATerm = (1.0 + Adelta)/A->sigma;
        if (CONCURRENT) { A->unlock(); }

        ND::predTY & Apreds = A->preds;
        int sz = Apreds.size();
        for (int i=0; i<sz; ++i) {
          ND *pd = Apreds[i];
          //const double term = pd->sigma*ATerm; 
          const double term = pd->sigma*(1.0 + Adelta)/A->sigma; 
          if (CONCURRENT) { pd->lock(); }
          pd->delta += term;
          const int prevPdNsuccs = pd->nsuccs;
          pd->nsuccs--;

          if (prevPdNsuccs == 1) {
            if (CONCURRENT) { pd->unlock(); }
            ctx.push(pd);
          } else {
            if (CONCURRENT) { pd->unlock(); }
          }
        }
        A->reset();
        graph->resetOutEdges(A);
      } else {
        galois::gDebug("Skipped ", A->toString());
        if (CONCURRENT) { A->unlock(); }
      }
    }
};

std::vector<std::pair<int, int>> nodeArrayRanges;
std::vector<std::pair<int, int>> edgeArrayRanges;
std::vector<int> workChunks;
void createCleanupChunks(int nnodes, int nedges, int numThreads) {
  int nChunkSize = nnodes / numThreads;
  int eChunkSize = nedges / (numThreads);

  galois::gDebug("nChunkSize: ", nChunkSize, " eChunkSize: ", eChunkSize, 
                 " nnodes: ", nnodes, " nedges: ", nedges, " numThreads: ", 
                 numThreads);

  for (int i=0; i<numThreads; ++i) {
    int start = nChunkSize * i;
    int end = -1;
    if (i==numThreads-1)
      end = std::max(start+nChunkSize, nnodes);
    else
      end = std::min(start+nChunkSize, nnodes);
    galois::gDebug("Node cleanup chunk: ", i, " start: ", start, " end: ", end);
    nodeArrayRanges.push_back(std::make_pair(start, end));
    start = eChunkSize * i;
    if (i==numThreads-1)
      end = std::max(start+eChunkSize, nedges);
    else
      end = std::min(start+eChunkSize, nedges);
    edgeArrayRanges.push_back(std::make_pair(start, end));
    galois::gDebug("Edge cleanup chunk: ", i, " start: ", start, " end: ", end);
    workChunks.push_back(i);
  }
}


galois::InsertBag<ND*>* fringewl;
galois::substrate::CacheLineStorage<ND> *gnodes;

struct fringeFindDOALL2NoInline {
  void operator()(int i,int) const {
    std::pair<int,int> p1 = nodeArrayRanges[i];
    int start = p1.first;
    int end = p1.second;
    for (int j=start; j<end; ++j)
    operator()(j);
  }

  void operator()(int j) const {
    ND * n = &(gnodes[j].data);
    if (n->nsuccs == 0 && n->distance < DEF_DISTANCE) {
      fringeCnts.update(1);
          
      fringewl->push(n);
    }
  }
};

struct NodeIndexer : std::binary_function<ND*, int, int> {
  int operator() (const ND *val) const {
  //  return val->level / buckets;
    return val->distance /*/ buckets*/;
  }
};

static const char* name = "Betweenness Centrality";
static const char* desc = "Computes betwenness centrality in an unweighted "
                          "graph";

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, NULL);

  if (CONCURRENT) {
    galois::gInfo("Running in concurrent mode with ", numThreads);
  } else {
    galois::gInfo("Running in serial mode");
  }

  graph = new BCGraph(filename.c_str());
  unsigned nnodes = graph->size();
  unsigned nedges = graph->getNedges();
  galois::gInfo("Num nodes is ", nnodes, ", num edges is ", nedges);

  DEF_DISTANCE = nnodes * 2;
  
  galois::gInfo("Bucket size ", buckets);
  galois::gInfo("Num threads is ", numThreads);

  createCleanupChunks(nnodes, nedges, numThreads);

  gnodes = graph->getNodes();
  firstForEachNodeBased feach1NodeBased;
  secondForEach feach2;
  fringeFindDOALL2NoInline findFringe;

  
  galois::StatTimer initCapTimer("InitCapacities");
  initCapTimer.start();
  graph->fixNodePredsCapacities();
  initCapTimer.stop();

  action1cnt.reset();
  action2cnt.reset();
  action2Excnt.reset();
  action3cnt.reset();
  action4cnt.reset();
  action555.reset();
  actionNone.reset();
  largestNodeDist.reset();

  #if CONCURRENT
  // FOR RMAT25, RAND26 55
  #ifdef USE_NODE_BASED
  // for rmat25 chz = 4, dcl, back : cdl16
  // new regime: rmat25 dcf4, dcl8  //same dcl8, dcl16
  // new regime: rand26 dcl4, dcl16, ///better dcl32, dcl16
  const int chunksize = 8;
  galois::gInfo("Using chunk size : ", chunksize);
  typedef galois::worklists::OrderedByIntegerMetric<NodeIndexer, 
                                                    galois::worklists::dChunkedLIFO<chunksize> > wl2ty;
  //typedef Galoisruntime::worklists::ChunkedFIFO<chunksize, ND*, true> wl2ty;
  galois::InsertBag<ND*> wl2;
  #else
  //const int chunksize = 64;
  //std::cerr << "Using chunk size : " << chunksize << "\n";
  //Galoisruntime::galois_insert_bag<ED*> wl2;
  #endif
  galois::InsertBag<ND*> wl4;
  fringewl = &wl4;
  //Galoisruntime::worklists::FIFO<int,  true> wl5;
  #else 
  // not CONCURRENT
  galois::worklists::GFIFO<ED*, false> wl2;
  galois::worklists::GFIFO<ND*, false> wl4;
  #endif 

  galois::reportPageAlloc("MemAllocPre");

  unsigned stepCnt = 0; // number of sources done

  galois::StatTimer executionTimer;
  galois::StatTimer firstLoopTimer("First Loop");
  galois::StatTimer secondLoopTimer("Second Loop");
  galois::StatTimer thirdLoopTimer("Third Loop");
  galois::StatTimer fourthLoopTimer("Fourth Loop");

  executionTimer.start();
  for (unsigned i = startNode; i < nnodes; ++i) {
    ND* active = &(gnodes[i].data);
    currSrcNode = active;
    // ignore nodes with no neighbors
    int nnbrs = graph->outNeighborsSize(active);
    if (nnbrs == 0) {
      continue;
    }

    galois::do_all(
      galois::iterate(0u, nnodes),
      [&] (auto j) {
        if (j != i) {
          (gnodes[j].data).reset();
        }
      }
    );
    
    stepCnt++;
    //if (stepCnt >= 2) break;  // ie only do 1 source

    //std::list<ED*> wl;
#ifdef USE_NODE_BASED
         std::vector<ND*>  wl;
    wl2.push_back(active);
#else
    std::vector<ED*> wl;
    graph->initWLToOutEdges(active, wl2);
#endif
//    wl2.fill_initial(wl.begin(), wl.end());
    active->initAsSource();
    galois::gDebug("Source is ", active->toString());
    //graph->checkClearEdges();
//__itt_resume();
    firstLoopTimer.start();
#ifdef USE_NODE_BASED

#if CONCURRENT
    galois::for_each(galois::iterate(wl2.begin(), wl2.end()), 
                     feach1NodeBased, galois::wl<wl2ty>());
    //galois::for_each(galois::iterate(wl2.begin(), wl2.end()), 
    //                 feach1NodeBased, galois::wl<wl2ty>());
    wl2.clear();
//    galois::for_each<wl2ty>(active, feach1NodeBased);
#else
  Timer tt;
    tt.start();

  while (!wl2.empty()) {
      Nd *nn = wl2.pop().second;
      feach1NodeBased(nn, wl2);
    }
  tt.stop();
  galois::gInfo("tt ", tt.get());

#endif

#else

#if CONCURRENT
    galois::Timer firstLT;
    firstLT.start();
    galois::for_each(wl2.begin(), wl2.end(), feach1, galois::wl<wl2ty>());
    firstLT.stop();
    galois::gInfo("FLTime: ", firstLT.get());
#else
  Timer tt;
    tt.start();

  while (!wl2.empty()) {
      ED *ee = wl2.pop().second;
      feach1(ee, wl2);
    }
  tt.stop();
  galois::gInfo("tt ", tt.get());

#endif

#endif
    firstLoopTimer.stop();
//__itt_pause();
    if (DOCHECKS) graph->checkGraph(active);

    secondLoopTimer.start();
#if CONCURRENT

  fringeCnts.reset();

  galois::do_all(galois::iterate(0u, nnodes), findFringe);
  unsigned int fringeCnt = fringeCnts.reduce();
  fringeCnts.reset();
#else
    //std::list<ND*> wl3;
    for (int j=0; j<nnodes; ++j) {
      ND * n = &(gnodes[j].data);
      if (n->nsuccs == 0) {
        if (fringeCnts.isActive()) {
          fringeCnt++;
        }
        wl4.push_back(n);
      }
    }
    //wl4.fill_initial(wl3.begin(), wl3.end());
#endif
    secondLoopTimer.stop();

    if (fringeCnts.isActive()) {
      galois::gPrint(fringeCnt, " nodes in fringe");
    }

    thirdLoopTimer.start();
    double backupSrcBC = currSrcNode->bc;
    #if CONCURRENT   
    galois::for_each(galois::iterate(wl4), feach2);
    #else
    while (!wl4.empty()) {
      ND *nn = wl4.pop().second;
      feach2(nn, wl4);
    }
#endif
    currSrcNode->bc = backupSrcBC;
    thirdLoopTimer.stop();
    wl4.clear();
    if (DOCHECKS) graph->checkSteadyState2();
    //graph->printGraph();

    fourthLoopTimer.start();
    graph->cleanupData();
    fourthLoopTimer.stop();

  }
  executionTimer.stop();

  galois::reportPageAlloc("MemAllocPost");

  // one counter active -> all of them are active (since all controlled by same
  // ifdef)
  if (action1cnt.isActive()) {
    unsigned long sum1 = action1cnt.reduce();
    unsigned long sum2 = action2cnt.reduce();
    unsigned long sum2ex = action2Excnt.reduce();
    unsigned long sum3 = action3cnt.reduce();
    unsigned long sum4 = action4cnt.reduce();
    unsigned long sum555 = action555.reduce();
    unsigned long sumNone = actionNone.reduce();

    galois::gPrint("Action 1 ", sum1, "\nAction 2 ", sum2, 
                   "\nRealActionUS ", sum2ex, "\nAction 3 ", sum3, 
                   "\nAction 4 ", sum4, "\nAction4_mut ", sum555, 
                   "\nActionNone ", sumNone, "\n");
    galois::gPrint("Largest node distance is ", largestNodeDist.reduce(), "\n");
  }

  // prints out first 10 node BC values
  if (!skipVerify) {
    //graph->verify(); // TODO see what this does
    int count = 0;
    for (unsigned i = 0; i < nnodes && count < 10; ++i, ++count) {
      galois::gPrint(count, ": ", std::setiosflags(std::ios::fixed), 
                     std::setprecision(6), gnodes[i].data.bc, "\n");
    }
  }

  if (generateCert) {
    graph->printAllBCs(numThreads, "certificate_");
  }

  return 0;
}
