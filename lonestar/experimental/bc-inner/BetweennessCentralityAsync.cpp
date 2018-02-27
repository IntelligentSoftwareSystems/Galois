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
Counter spfuCount;
Counter updateSigmaP1Count;
Counter updateSigmaP2Count;
Counter firstUpdateCount;
Counter correctNodeP1Count;
Counter correctNodeP2Count;
Counter noActionCount;

using MaxCounter = 
  ConditionalAccumulator<galois::GReduceMax<unsigned long>, COUNT_ACTIONS>;
MaxCounter largestNodeDist;

using LeafCounter = 
  ConditionalAccumulator<galois::GAccumulator<unsigned long>, COUNT_LEAVES>;
LeafCounter leafCount;

struct ForwardPass {
  template<typename Context>
  void operator()(ND* srcD, Context& ctx) {
    int idx = srcD->id;
    #if USE_MARKING
    srcD->markOut();
    #endif

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
      ND* A = srcD;
      const int ADist = srcD->distance;
      ND* B = dstD;
      const int BDist = dstD->distance;

      // Shortest Path + First Update
      if (BDist - ADist > 1) {
        spfuCount.update(1);

        // make B a successor of A, A predecessor of B
        A->nsuccs++;
        const double ASigma = A->sigma;
        if (CONCURRENT) { A->unlock(); }
        ND::predTY & Bpreds = B->preds;
        bool bpredsNotEmpty = !Bpreds.empty();
        Bpreds.clear();
        Bpreds.push_back(A);
        B->distance = ADist + 1;

        B->nsuccs = 0; // SP
        B->sigma = ASigma; // FU
        ed.val = ASigma;
        ed.level = ADist;
        if (CONCURRENT) { B->unlock(); }

        #if USE_MARKING          
        if (!B->isAlreadyIn())
        #endif
          ctx.push(B);

        // Part of Correct Node
        if (bpredsNotEmpty) {
          int idx = B->id;
          int* inIdx = graph->inIdx;
          int* ins = graph->ins; 
          int startInE = inIdx[idx];
          int endInE = inIdx[idx + 1];

          // loop through in edges
          for (int j = startInE; j < endInE; j++) {
            ED & inE = edgeData[ins[j]];
            ND *inNbr = inE.src;
            if (inNbr == B) continue;

            // TODO wrap in concurrent
            // lock in right order
            ND* aL;
            ND* aW;
            if (B < inNbr) { aL = B; aW = inNbr;} 
            else { aL = inNbr; aW = B; }
            aL->lock(); aW->lock();

            const int elev = inE.level; 
            // Correct Node
            if (inNbr->distance >= B->distance) { 
              correctNodeP1Count.update(1);
              if (CONCURRENT) { B->unlock(); }

              galois::gDebug("Rule 4 (", inNbr->toString(), " ||| ", 
                             B->toString(), ")", elevel);

              if (elev != DEF_DISTANCE) {
                inE.level = DEF_DISTANCE;
                if (elev == inNbr->distance) {
                  correctNodeP2Count.update(1);
                  inNbr->nsuccs--;
                }
              }
              if (CONCURRENT) { inNbr->unlock(); }
            } else {
              aL->unlock(); aW->unlock();
            }
          }
        }
      // Update Sigma
      } else if (elevel == ADist && BDist == ADist + 1) {
        updateSigmaP1Count.update(1);
        galois::gDebug("Rule 2 (", A->toString(), " ||| ", B->toString(), ") ",
                       elevel);
        const double ASigma = A->sigma;
        const double eval = ed.val;
        const double diff = ASigma - eval;
        bool BSigmaChanged = diff >= 0.00001;
        if (CONCURRENT) { A->unlock(); }
        if (BSigmaChanged) {
          updateSigmaP2Count.update(1);
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
      // First Update not combined with Shortest Path
      } else if (BDist == ADist + 1 && elevel != ADist) {
        firstUpdateCount.update(1);
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
      } else { // No Action
        noActionCount.update(1);
        A->unlock(); B->unlock();
      }
    }
  }    
};

struct BackwardPass {
  template<typename Context>
  void operator()(ND* A, Context& ctx) {
    if (CONCURRENT) { A->lock(); }

    if (A->nsuccs == 0) {
      const double Adelta = A->delta;
      A->bc += Adelta;

      if (CONCURRENT) { A->unlock(); }

      ND::predTY& Apreds = A->preds;
      int sz = Apreds.size();

      // loop through A's predecessors
      for (int i = 0; i < sz; ++i) {
        ND* pd = Apreds[i];
        const double term = pd->sigma * (1.0 + Adelta) / A->sigma; 
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

struct FindLeaves {
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
      leafCount.update(1);
          
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
  ForwardPass dagConstruction;
  FindLeaves leafFinder;
  BackwardPass dependencyBackProp;
  
  galois::StatTimer initCapTimer("InitCapacities");
  initCapTimer.start();
  graph->fixNodePredsCapacities();
  initCapTimer.stop();

  spfuCount.reset();
  updateSigmaP1Count.reset();
  updateSigmaP2Count.reset();
  firstUpdateCount.reset();
  correctNodeP1Count.reset();
  correctNodeP2Count.reset();
  noActionCount.reset();
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
  galois::StatTimer forwardPassTimer("ForwardPass");
  galois::StatTimer leafFinderTimer("LeafFind");
  galois::StatTimer backwardPassTimer("BackwardPass");
  galois::StatTimer cleanupTimer("CleanupTimer");

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
    forwardPassTimer.start();
#ifdef USE_NODE_BASED

#if CONCURRENT
    galois::for_each(galois::iterate(wl2), dagConstruction, galois::wl<wl2ty>());
    wl2.clear();
#else
  Timer tt;
  tt.start();

  while (!wl2.empty()) {
    Nd *nn = wl2.pop().second;
    dagConstruction(nn, wl2);
  }
  tt.stop();
  galois::gInfo("tt ", tt.get());
#endif

#else
// EDGE BASED
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
    forwardPassTimer.stop();
//__itt_pause();
    if (DOCHECKS) graph->checkGraph(active);

    leafFinderTimer.start();
#if CONCURRENT
  leafCount.reset();
  galois::do_all(galois::iterate(0u, nnodes), leafFinder);
#else
    //std::list<ND*> wl3;
    for (int j=0; j<nnodes; ++j) {
      ND * n = &(gnodes[j].data);
      if (n->nsuccs == 0) {
        // TODO technically to not use galois struct use regular counter
        leafCount.update(1);
        wl4.push_back(n);
      }
    }
    //wl4.fill_initial(wl3.begin(), wl3.end());
#endif
    leafFinderTimer.stop();

    if (leafCount.isActive()) {
      galois::gPrint(leafCount.reduce(), " leaf nodes in DAG");
    }

    backwardPassTimer.start();
    double backupSrcBC = currSrcNode->bc;
    #if CONCURRENT   
    galois::for_each(galois::iterate(wl4), dependencyBackProp);
    #else
    while (!wl4.empty()) {
      ND *nn = wl4.pop().second;
      dependencyBackProp(nn, wl4);
    }
    #endif
    currSrcNode->bc = backupSrcBC; // current source BC should not get updated
    backwardPassTimer.stop();
    wl4.clear();
    if (DOCHECKS) graph->checkSteadyState2();
    //graph->printGraph();

    cleanupTimer.start();
    graph->cleanupData();
    cleanupTimer.stop();
  }
  executionTimer.stop();

  galois::reportPageAlloc("MemAllocPost");

  // one counter active -> all of them are active (since all controlled by same
  // ifdef)
  if (spfuCount.isActive()) {
    galois::gPrint("SP&FU ", spfuCount.reduce(), 
                   "\nUpdateSigmaBefore ", updateSigmaP1Count.reduce(), 
                   "\nRealUS ", updateSigmaP2Count.reduce(), 
                   "\nFirst Update ", firstUpdateCount.reduce(), 
                   "\nCorrectNodeBefore ", correctNodeP1Count.reduce(), 
                   "\nReal CN ", correctNodeP2Count.reduce(), 
                   "\nNoAction ", noActionCount.reduce(), "\n");
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
