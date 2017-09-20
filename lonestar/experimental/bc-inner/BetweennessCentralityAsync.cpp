#include "Galois/Galois.h"
#include "Galois/Timer.h"
#include "Galois/Timer.h"
#include "Galois/Bag.h"
#include "Galois/Substrate/CacheLineStorage.h"
#include "Lonestar/BoilerPlate.h"

#include <boost/tuple/tuple.hpp>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <list>
#include <algorithm>
#include <utility>

#include "control.h"
#include "ND.h"
#include "ED.h"
#include "BCGraph.h"
#include "util.h"

#define BUCKETSZ 1 
#define GENERATE_CERTIFICATE 0

#define DBG_FRINGECNT 0

#define USE_NODE_BASED 1
//#define INLINE_US 1

using namespace std;

//static const char* help = "<input file>";

int DEF_DISTANCE;

BCGraph *  graph;
ND * currSrcNode;

//#define COUNT_ACTIONS 1
#ifdef COUNT_ACTIONS
PerCPU<unsigned long> action1cnt ;
PerCPU<unsigned long> action2cnt ;
PerCPU<unsigned long> action2Excnt ;
PerCPU<unsigned long> action3cnt ;
PerCPU<unsigned long > action4cnt ;
PerCPU<unsigned long> action555 ;
PerCPU<unsigned long> actionNone;
PerCPU<unsigned int> largestNodeDist;
#endif

#if MERGE_LOOPS
int curr_round = 0;
#endif

struct f2Item {
//  public:
  bool isCleanup;
  ND *node;
  double d;

//  f2Item(bool is, ND *nd, double _d) : isCleanup(is), node(nd), d(_d) {}
};

struct firstForEachNodeBased {

  typedef int tt_does_not_need_stats;
  typedef int tt_does_not_need_aborts;

  template<typename Context>
    void  inline /*__attribute__((noinline))*/ operator()(ND* srcD, Context& ctx) {

      int idx = srcD->id;
#if USE_MARKING
      srcD->markOut();
#endif
//      assert(idx <= nnodes-1);
      int * outIdx = graph->outIdx;
      int startE = outIdx[idx];
      int endE = outIdx[idx + 1];
      ED * edgeData = graph->edgeData;
      for (int i = startE; i < endE; i++) {
        ED & ed = edgeData[i];
        ND * dstD = ed.dst;
        
        if (srcD == dstD) continue;

        if (CONCURRENT) {
          ND *loser, *winner;
          if (srcD < dstD) { loser = srcD; winner = dstD;} 
          else { loser = dstD; winner = srcD; }
          loser->lock();
          winner->lock();
        }
        const int elevel = ed.level;
        ND * A = srcD; const int ADist = srcD->distance;
        ND * B = dstD; const int BDist = dstD->distance;

        if (BDist - ADist > 1) { // Rule 1 + Rule 3 combined
#if COUNT_ACTIONS
          action1cnt.get()++;
#endif
          if (DBG) { cerr << "Rule 1+3 (" << A->toString() << " ||| " << B->toString() << ") " << elevel << endl; }
          A->nsuccs++;
          const double ASigma = A->sigma;
          if (CONCURRENT) { A->unlock(); }
          ND::predTY & Bpreds = B->preds;
          bool bpredsNotEmpty = !Bpreds.empty();
          //if (bpredsNotEmpty) std::cerr << Bpreds.size() << std::endl;
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
#if COUNT_ACTIONS
              action4cnt.get()++;
#endif
              if (inNbr->distance >= B->distance) { // Rule 4
                if (CONCURRENT) { B->unlock(); }
                if (DBG) cerr << "Rule 4 (" << inNbr->toString() << " ||| " << B->toString() << ")" << elevel << endl;
                if (elev != DEF_DISTANCE) {
                  inE.level = DEF_DISTANCE;
                  if (elev == inNbr->distance) {
#if COUNT_ACTIONS
                    action555.get()++;
#endif
                    inNbr->nsuccs--;
                  }
                }
                if (CONCURRENT) { inNbr->unlock(); }
              } else { aL->unlock(); aW->unlock(); }
            }
          }
        } else if (elevel == ADist && BDist == ADist + 1) { // Rule 2: BDist = ADist + 1 and elevel = ADist
#if COUNT_ACTIONS
          action2cnt.get()++;
#endif
          if (DBG) { cerr << "Rule 2 (" << A->toString() << " ||| " << B->toString() << ") " << elevel << endl; }
          const double ASigma = A->sigma;
          const double eval = ed.val;
          const double diff = ASigma - eval;
          bool BSigmaChanged = diff >= 0.00001;
          if (CONCURRENT) { A->unlock(); }
          if (BSigmaChanged) {
#if COUNT_ACTIONS
          action2Excnt.get()++;
#endif
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
                  if (DBG) { cerr << "Rule 2 (" << A->toString() << " ||| " << B->toString() << ") " << elevel << endl; }
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
#if COUNT_ACTIONS
          action3cnt.get()++;
#endif 
          // Test taking it out
          //const bool AnotPredOfB = !B->predsContain(A);
          //if (!AnotPredOfB) {
          //	cerr << "I DONT UNDERSTANT" << A->toString() << " " << B->toString() << endl;
          //}
          //if (AnotPredOfB) {
          A->nsuccs++;
          //}
          const double ASigma = A->sigma;
          if (CONCURRENT) { A->unlock(); }

          //if (AnotPredOfB) {
          B->preds.push_back(A);
          //}
          const double BSigma = B->sigma;
          if (DBG) { cerr << "Rule 3 (" << A->toString() << " ||| " << B->toString() << ") " << elevel << endl; }
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
                  if (DBG) { cerr << "Rule 2 (" << A->toString() << " ||| " << B->toString() << ") " << elevel << endl; }
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
        } /*else if (ADist >= BDist) { // Rule 4
          if (CONCURRENT) { B->unlock(); }
          if (DBG) cerr << "Rule 4 (" << A->toString() << " ||| " << B->toString() << ")" << elevel << endl;
          if (elevel != DEF_DISTANCE) { 
            ed.level = DEF_DISTANCE;                 
            if (elevel == ADist
#if MERGE_LOOPS               
                && ed.cnt == curr_round
#endif                     
                ) { 
                    A->nsuccs--;
#if MERGE_LOOPS                    
                    ed.cnt = curr_round;
#endif  
            }  
          }      
          if (CONCURRENT) { A->unlock(); }              
        }*/ else {
#if COUNT_ACTIONS
          actionNone.get()++;
#endif
          A->unlock(); B->unlock();
        }
      }
    }    
};

struct firstForEach {
	template<typename Context>
	void  inline /*__attribute__((noinline))*/ operator()(ED* ed, Context& ctx) {
    //itersOfFirstBody++;
		ND * srcD = ed->src;
		ND * dstD = ed->dst;
//		ed->markOut();
		
    
    assert(srcD->id >= 0 && srcD->id < graph->size());
    assert(dstD->id >= 0 && dstD->id < graph->size());

    if (CONCURRENT) {
			ND *loser, *winner;
			if (srcD/*->id*/ < dstD/*->id*/) {
				//srcD->lock();
				//dstD->lock();
				loser = srcD;
				winner = dstD;
			} else {
				loser = dstD;
				winner = srcD;
				//dstD->lock();
				//srcD->lock();
			}
			loser->lock();
			winner->lock();
		}
		//if (DBG) System.err.println("Extracted " + srcD + "bbbbb " + dstD);
		const int srcdist = srcD->distance;
		const int dstdist = dstD->distance;
		const int elevel = ed->level;
    
		//System.err.println(srcD + " |||| " + dstD );

    if (srcdist >= dstdist) { // Rule 4
      if (CONCURRENT) { dstD->unlock(); }
#ifdef COUNT_ACTIONS
		action4cnt.get()++;
#endif
      if (DBG) cerr << "Rule 4 (" << srcD->toString() << " ||| " << dstD->toString() << ")" << elevel << endl;
      if (elevel != DEF_DISTANCE /*&& elevel == srcdist*/) {
        ed->level = DEF_DISTANCE;
        if (elevel == srcdist 
#if MERGE_LOOPS 
						&& ed->cnt == curr_round
#endif
						) { 
          srcD->nsuccs--;
#ifdef COUNT_ACTIONS
//        std::cerr << "Got one \n";
        action555.get()++;
#endif

#if MERGE_LOOPS
					ed->cnt = curr_round;
#endif
				}
      }
      //ed->level = DEF_DISTANCE;
      if (CONCURRENT) { srcD->unlock(); }
      return;
    }

		// From this point on srcdist < dstdist, set A to be the
		// lower level node and B to be the higher level node
		ND * A = srcD; const int ADist = srcdist;
		ND * B = dstD; const int BDist = dstdist;
		// That is BDist >= ADist + 1

    if (BDist - ADist >= 2) { // Rule 1 + Rule 3 combined
#ifdef COUNT_ACTIONS
		action1cnt.get()++;
#endif
      if (DBG) { cerr << "Rule 1+3 (" << A->toString() << " ||| " << B->toString() << ") " << elevel << endl; }
      A->nsuccs++;
      const double ASigma = A->sigma;
      if (CONCURRENT) { A->unlock(); }
			
	  		ND::predTY & Bpreds = B->preds;
	  		bool bpredsNotEmpty = !Bpreds.empty();
			//Bpreds.clear();
			Bpreds.clear();//resize(0);
			Bpreds.push_back(A);
			B->distance = ADist + 1;
			//int newBDist = ADist + 1;
#ifdef COUNT_ACTIONS
			if (BDist > largestNodeDist.get())
				largestNodeDist.get() = B->distance;
#endif
			B->nsuccs = 0;
			B->sigma = ASigma;
			ed->val = ASigma;
			ed->level = ADist;
#if MERGE_LOOPS
			ed->cnt = curr_round;
#endif
			if (CONCURRENT) { B->unlock(); }
			//if (BDist != Util.infinity)
			if (bpredsNotEmpty) 
				graph->addInEdgesToWL(B,/* BDist,*/ ctx, ed);
			//addInEdgesToWL(B, ed, ctx);
			//addInNeighborsToWL(B, A, ctx);
			graph->addOutEdgesToWL(B, /*newBDist,*/ ctx);
			return;
    } else if (elevel == ADist
#if MERGE_LOOPS
				&& ed->cnt == curr_round
#endif
				) { // Rule 2: BDist = ADist + 1 and elevel = ADist
#ifdef COUNT_ACTIONS
      action2cnt.get()++;
#endif
      if (DBG) { cerr << "Rule 2 (" << A->toString() << " ||| " << B->toString() << ") " << elevel << endl; }
      const double ASigma = A->sigma;
      const double eval = ed->val;
      const double diff = ASigma - eval;
			bool BSigmaChanged = diff >= 0.00001;
			if (CONCURRENT) { A->unlock(); }
			if (BSigmaChanged) {
				ed->val = ASigma;
#if MERGE_LOOPS
			  ed->cnt = curr_round;
#endif
				B->sigma += diff;
				int nbsuccs = B->nsuccs;
				if (CONCURRENT) { B->unlock(); }
				if (nbsuccs > 0)
					graph->addOutEdgesToWL(B, /*BDist,*/ ctx);
			} else {
				//if (CONCURRENT) { A->unlock(); }
				if (CONCURRENT) { B->unlock(); }
			}
			return;
		} else {  // Rule 3 // BDist = ADist + 1 and elevel != ADist
#ifdef COUNT_ACTIONS
			action3cnt.get()++;
#endif
			// Test taking it out
			//const bool AnotPredOfB = !B->predsContain(A);
			//if (!AnotPredOfB) {
			//	cerr << "I DONT UNDERSTANT" << A->toString() << " " << B->toString() << endl;
			//}
			//if (AnotPredOfB) {
				A->nsuccs++;
			//}
			const double ASigma = A->sigma;
			if (CONCURRENT) { A->unlock(); }

			//if (AnotPredOfB) {
				B->preds.push_back(A);
			//}
			const double BSigma = B->sigma;
      if (DBG) { cerr << "Rule 3 (" << A->toString() << " ||| " << B->toString() << ") " << elevel << endl; }
      B->sigma = BSigma + ASigma;
			ed->val = ASigma;
			ed->level = ADist;
#if MERGE_LOOPS
			ed->cnt = curr_round;
#endif
			int nbsuccs = B->nsuccs;
			if (CONCURRENT) { B->unlock(); }
			const bool BSigmaChanged = ASigma >= 0.00001;
			if (nbsuccs > 0 && BSigmaChanged) {
				graph->addOutEdgesToWL(B, /*BDist,*/ ctx);
			}
			return;
		}
	}
};

//namespace galois {
//  template<>
//    struct does_not_need_aborts<firstForEach> : public boost::true_type {};
//}

struct secondForEach {

  typedef int tt_does_not_need_stats;
  typedef int tt_does_not_need_aborts;

  template<typename Context>
    void inline /*__attribute__((noinline))*/ operator()(ND *A, Context& ctx) {
      if (CONCURRENT) { A->lock(); }
      if (A->nsuccs == 0 /*&& !A->deltaDone()*/) {
        //...A->nsuccs = -1;
        //A->setDeltaDoneT();//A.deltaDone = true;
        if (DBG) { std::cerr << "RULE 1 " << A->toString() << std::endl; }
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
/*          if (!(prevPdNsuccs >= 1)) { 
            std::cerr << "BUG: " << pd->toString() << " " << (prevPdNsuccs-1) << std::endl;
            assert(false);
          }*/
          if (prevPdNsuccs == 1) {
            if (CONCURRENT) { pd->unlock(); }
            ctx.push(pd);
          } else {
            if (CONCURRENT) { pd->unlock(); }
          }
        }
				A->reset();
        graph->resetOutEdges(A);
//        if (A == currSrcNode) {
//          std::cerr << "Clean src\n";
//        }

      } else {
        /*if (DBG)*/ { std::cerr << "Skipped " << A->toString() << endl; }
        if (CONCURRENT) { A->unlock(); }
      }
    }
};

struct secondForEachWithInline {

  typedef int tt_does_not_need_stats;
  typedef int tt_does_not_need_aborts;

  template<typename Context>
    void inline operator()(f2Item tt, Context& ctx) {
      ND * A = tt.node;
      if (tt.isCleanup) {
        graph->resetOutEdges(A);
        return;
      }
      const double Adelta = tt.d;
//      if (CONCURRENT) { A->lock(); }
//      if (A->nsuccs == 0) {
        if (DBG) { std::cerr << "RULE 1 " << A->toString() << std::endl; }
//        const double Adelta = A->delta;
        A->bc += Adelta;
//        if (CONCURRENT) { A->unlock(); }

        ND::predTY & Apreds = A->preds;
        int sz = Apreds.size();
        for (int i=0; i<sz; ++i) {
          ND *pd = Apreds[i];
          const double term = pd->sigma*(1.0 + Adelta)/A->sigma; 
          if (CONCURRENT) { pd->lock(); }
          pd->delta += term;
          const int prevPdNsuccs = pd->nsuccs;
          pd->nsuccs--;
          if (prevPdNsuccs == 1) {
            const double pdDelta = pd->delta;
            pd->bc += pdDelta;
            if (CONCURRENT) { pd->unlock(); }
            ND::predTY & pdPreds = pd->preds;
            int sz = pdPreds.size();
            for (int j=0; j<sz; ++j) {
              ND *pd2 = pdPreds[j];
              const double term = pd2->sigma*(1.0 + pdDelta)/pd->sigma; 
              if (CONCURRENT) { pd2->lock(); }
              pd2->delta += term;
              const int prevPd2Nsuccs = pd2->nsuccs;
              pd2->nsuccs--;
              if (prevPd2Nsuccs == 1) {
                const double pd2Delta = pd2->delta;
                pd2->bc += pd2Delta;
                if (CONCURRENT) { pd2->unlock(); }
                ND::predTY & pd2Preds = pd2->preds;
                int sz = pd2Preds.size();
                for (int k=0; k<sz; ++k) {
                  ND *pd3 = pd2Preds[k];
                  const double term = pd3->sigma*(1.0 + pd2Delta)/pd2->sigma; 
                  if (CONCURRENT) { pd3->lock(); }
                  pd3->delta += term;
                  double pd3Delta = pd3->delta;
                  const int prevPd3Nsuccs = pd3->nsuccs;
                  pd3->nsuccs--;               
                  if (CONCURRENT) { pd3->unlock(); }
                  if (prevPd3Nsuccs == 1) {
                    f2Item f = { false,pd3,pd3Delta};
                    //ctx.push(f2Item(false,pd3,pd3Delta));
                    ctx.push(f);
                  } 
                }
				        pd2->reset();
                //ctx.push(f2Item(true,pd2,0));
                f2Item f = {true,pd2,0};
                ctx.push(f);
                //ctx.push(f2Item(true,pd2,0));
                //graph->resetOutEdges(pd2);
              } else {
                if (CONCURRENT) { pd2->unlock(); }
              }
            }
				    pd->reset();
            //ctx.push(f2Item(true,pd,0));
            f2Item f = {true,pd,0};
            ctx.push(f);
            //graph->resetOutEdges(pd);
            //ctx.push(pd);            
          } else {
            if (CONCURRENT) { pd->unlock(); }
          }
        }
				A->reset();
        graph->resetOutEdges(A);

//      } else {
//        /*if (DBG)*/ { std::cerr << "Skipped " << A->toString() << endl; }
//        if (CONCURRENT) { A->unlock(); }
//      }
    }
};





//namespace galois {
//  template<>
//    struct does_not_need_aborts<secondForEach> : public boost::true_type {};
//}


std::vector<std::pair<int,int> > nodeArrayRanges;
std::vector<std::pair<int,int> > edgeArrayRanges;
std::vector<int> workChunks;
void createCleanupChunks(int nnodes, int nedges, int numThreads) {
  int nChunkSize = nnodes / numThreads;
  int eChunkSize = nedges / (numThreads);

  //if (DBG) {
  	cerr << "nChunkSize: " << nChunkSize << " EChunkSize: " << eChunkSize << " nnodes: " << nnodes << " nedges: " << nedges << " numThreads: " << numThreads << endl;
  //}
  for (int i=0; i<numThreads; ++i) {
    int start = nChunkSize * i;
    int end = -1;
    if (i==numThreads-1)
      end = max(start+nChunkSize, nnodes);
    else
      end = min(start+nChunkSize, nnodes);
	if (DBG) { cerr << "Node cleanup chunk: " << i << " start: " << start << " end: " << end << endl; }
    nodeArrayRanges.push_back(std::make_pair(start, end));
    start = eChunkSize * i;
    if (i==numThreads-1)
      end = max(start+eChunkSize, nedges);
    else
      end = min(start+eChunkSize, nedges);
    edgeArrayRanges.push_back(std::make_pair(start, end));
	if (DBG) { cerr << "Edge cleanup chunk: " << i << " start: " << start << " end: " << end << endl; }
    workChunks.push_back(i);
  }
}

struct RevNodeIndexer : std::binary_function<ND*, int, int> {
	int operator() (const ND *val) const {
		return 20-val->distance;
	}
};

#if DBG_FRINGECNT
PerCPU<int> fringeCnts;
#endif

//std::vector<f2Item> *fringeBuffs;
galois::InsertBag<ND*>* fringewl;
//GaloisRuntime::WorkList::SimpleOrderedByIntegerMetric<ND*, RevNodeIndexer, GaloisRuntime::WorkList::ChunkedFIFO<ND*, 256, true>, true> *fringewl;
galois::Substrate::CacheLineStorage<ND> *gnodes;

//void fringeFindOMP() {
//  int gsize = graph->size();
//#pragma omp parallel 
//  {
//    int lcnt = 0; 
//    int _tid_ = omp_get_thread_num();
//    #pragma omp for schedule(guided, 5)
//    for (lcnt=0; lcnt<gsize; ++lcnt) {
//      ND *n = graph->getNode(lcnt);
//			if (n->nsuccs == 0 && n->distance < DEF_DISTANCE) {
//#if DBG_FRINGECNT
//				fringeCnts.get()++;
//#endif
//        f2Item f = {false,n,n->delta};
//				fringeBuffs[_tid_].push_back(f);
////        fringewl->push(f2Item(false,n,n->delta));
//			}
//    }
//  }
//}

struct fringeFindDOALL {
	void inline operator()(int i,int) {
    std::pair<int,int> p1 = nodeArrayRanges[i];
		int start = p1.first;
		int end = p1.second;
		for (int j=start; j<end; ++j) {
			ND * n = &(gnodes[j].data);
			if (n->nsuccs == 0 && n->distance < DEF_DISTANCE) {
#if DBG_FRINGECNT
				fringeCnts.get()++;
#endif
//				fringeBuffs.get(i).push_back(f2Item(false,n,n->delta));
//        fringewl->push(f2Item(false,n,n->delta));

//        f2Item f = {false, n, n->delta};
//        fringewl->push(f);//2Item(false,n,n->delta));
			}
		}
	}
};

struct fringeFindDOALL2 {
	void inline operator()(int i,int) {
    std::pair<int,int> p1 = nodeArrayRanges[i];
		int start = p1.first;
		int end = p1.second;
		for (int j=start; j<end; ++j)
    operator()(j);
  }

  void operator()(int j)
    {
			ND * n = &(gnodes[j].data);
			if (n->nsuccs == 0 && n->distance < DEF_DISTANCE) {
#if DBG_FRINGECNT
				fringeCnts.get()++;
#endif
//				fringeBuffs.get(i).push_back(f2Item(false,n,n->delta));
//f2Item f = {false, n, n->delta};
//        fringewl->push(f);//2Item(false,n,n->delta));
			}
		}
};

struct fringeFindDOALL2NoInline {
  void operator()(int i,int) const {
    std::pair<int,int> p1 = nodeArrayRanges[i];
		int start = p1.first;
		int end = p1.second;
		for (int j=start; j<end; ++j)
    operator()(j);
  }

  void operator()(int j) const
    {
			ND * n = &(gnodes[j].data);
			if (n->nsuccs == 0 && n->distance < DEF_DISTANCE) {
#if DBG_FRINGECNT
				fringeCnts.get()++;
#endif
//				fringeBuffs.get(i).push_back(f2Item(false,n,n->delta));
    				
        fringewl->push(n);
			}
		}
};


struct initCapDOALL {
  void inline operator()(int i) {
    std::pair<int,int> p1 = nodeArrayRanges[i];
    int start = p1.first;
    int end = p1.second;
    for (int j=start; j<end; ++j) {
      ND * n = &(gnodes[j].data);
      int nOutNbrs = graph->inNeighborsSize(n);
      n->preds.reserve(nOutNbrs);
    }
  }
};

struct fourthForEach {
  template<typename Context>
    void /*__attribute__((noinline))*/ operator()(int i, Context& ctx) {
//      assert(i<nodeArrayRanges.size());
//      std::pair<int,int> p1 = nodeArrayRanges[i];
      assert(i<edgeArrayRanges.size());
      std::pair<int,int> p2 = edgeArrayRanges[i];
      //if (DBG) { cerr << "Cleaning up: nodes[" << p1.first << "," << p1.second << "] Edges[" << p2.first << "," << p2.second << "]" << endl; }
      graph->cleanupData(/*p1.first, p1.second,*/ p2.first, p2.second);
    }
};

struct cleanupGraphDOALL {
    void operator()(int i,int) {
//      assert(i<nodeArrayRanges.size());
//      std::pair<int,int> p1 = nodeArrayRanges[i];
      assert(i<(int)edgeArrayRanges.size());
      std::pair<int,int> p2 = edgeArrayRanges[i];
      //if (DBG) { cerr << "Cleaning up: nodes[" << p1.first << "," << p1.second << "] Edges[" << p2.first << "," << p2.second << "]" << endl; }
      graph->cleanupData(/*p1.first, p1.second,*/ p2.first, p2.second);
    }
};

struct EdgeIndexer : std::binary_function<ED*, int, int> {
	int operator() (const ED *val) const {
	//	return val->level / BUCKETSZ;
		return val->src->distance /*/ BUCKETSZ*/;
	}
};

struct NodeIndexer : std::binary_function<ND*, int, int> {
	int operator() (const ND *val) const {
	//	return val->level / BUCKETSZ;
		return val->distance /*/ BUCKETSZ*/;
	}
};
/*
struct f2ItemIndexer : std::binary_function<f2Item, int, int> {
	int operator() (const f2Item tt) const {
    return tt.pr;
	//	return val->level / BUCKETSZ;
	//	return val->distance / BUCKETSZ;
	}
};
*/

struct EdgeIndexer22 : std::binary_function<ED*, int, int> {
	int operator() (const ED *val) const {
		//return val->level / BUCKETSZ;
		return val->src->id / BUCKETSZ;
	}
};

struct EdgeIndexer2 : std::binary_function<ED*, int, int> {
	int operator() (const ED *val) const {
		//return -graph->inNeighborsSize(val->src);
		return val->src->distance < val->dst->distance ? 0 : 1;// / BUCKETSZ;
	}
};


// input: ("rmat"|"snap") <filename>

namespace cll = llvm::cl;
 
static cll::opt<std::string> mode(
    cll::Positional,
    cll::desc("Choose a graph type:"),
    cll::Required);
   

static cll::opt<std::string> filename(cll::Positional,
    cll::desc("<input file>"),
    cll::Required);

static cll::opt<unsigned int> startNode("startNode", cll::desc("Node to start search from"), cll::init(0));


static const char* name = "Betweenness Centrality";
static const char* desc = "Computes betwenness centrality in an unweighted graph";


int main(int argc, char** argv) {
   galois::StatManager statManager;
   LonestarStart(argc, argv, name, desc, NULL);

#if CONCURRENT
  cerr << "Running in concurrent mode " << numThreads << endl;
#else
  cerr << "Running in serial mode" << endl;
#endif
//	std::vector<const char*> args = parse_command_line(argc, argv, help);
  int nnodes, nedges; 
  if (std::string("bin").compare(mode) == 0) {
    graph = new BCGraph(filename.c_str());
    nnodes = graph->size();
    nedges = graph->getNedges();
    cerr << "NNODES " << nnodes << " NEDGES " << nedges << endl;
  } else {
    boost::tuple<int, int, map<int, set<int>*>*, map<int, set<int>*>*> p;
    if (std::string("rmat").compare(mode) == 0) {
      p = readGraph(filename.c_str());
    } else if (std::string("snap").compare(mode) == 0) {
      p = readSnapDirectedGraph(filename.c_str());
    } else if (std::string("rand").compare(mode) == 0) {
      p = readRandomGraph(filename.c_str());
    }
    map<int, set<int>*>* outnbrs = p.get<2>();
    map<int, set<int>*>* innbrs = p.get<3>();
    nnodes = p.get<0>();
    nedges = p.get<1>();
    graph = new BCGraph(nnodes, nedges, *outnbrs, *innbrs);
    freeData(outnbrs, innbrs);
  }
	
  cerr << "Bucket size: " << BUCKETSZ << endl;
//  galois::setMaxThreads(numThreads);
 	cerr << "numThreads " << numThreads << endl; 
	//omp_set_num_threads(numThreads);
  createCleanupChunks(nnodes, nedges, numThreads);

	gnodes = graph->getNodes();
  //firstForEach feach1;
  firstForEachNodeBased feach1NodeBased;
  //secondForEachWithInline feach2;
  secondForEach feach2;
  //fourthForEach feach4;
  //fringeFindDOALL2 findFringe;
  fringeFindDOALL2NoInline findFringe;
  //cleanupGraphDOALL cleanupGloop;
  //initCapDOALL setPredCapacitiesloop;

  int stepCnt = 0;
  galois::TimeAccumulator firstLoopTimer;
  galois::TimeAccumulator secondLoopTimer;
  galois::TimeAccumulator thirdLoopTimer;
  galois::TimeAccumulator fourthLoopTimer;
  galois::TimeAccumulator totalTimer;
  
	galois::TimeAccumulator intCapTimer;	
  intCapTimer.start();
	//#if CONCURRENT
	//	GaloisRuntime::for_all_parallel(setPredCapacitiesloop);
	//#else
	graph->fixNodePredsCapacities();
	//#endif	
	intCapTimer.stop();
	std::cerr << " INIT CAP : " << intCapTimer.get() << endl;

#ifdef COUNT_ACTIONS
  action1cnt.get() = 0; 
  action2cnt.get() = 0; 
  action2Excnt.get() = 0; 
  action3cnt.get() = 0; 
  action4cnt.get() = 0; 
  action555.get() = 0;
  actionNone.get() = 0;
  largestNodeDist.get() = 0;
#endif

#if CONCURRENT
  //GaloisRuntime::WorkList::OrderByIntegerMetric<ED*, EdgeIndexer, GaloisRuntime::WorkList::ChunkedFIFO<ED*, 16, true>, true> wl2;
  
  // FOR RMAT25, RAND26 55
#if USE_NODE_BASED
  // for rmat25 chz = 4, dcl, back : cdl16
  // new regime: rmat25 dcf4, dcl8  //same dcl8, dcl16
  // new regime: rand26 dcl4, dcl16, ///better dcl32, dcl16
  const int chunksize = 8;
  std::cerr << "Using chunk size : " << chunksize << std::endl;
  typedef galois::WorkList::OrderedByIntegerMetric<NodeIndexer, galois::WorkList::dChunkedLIFO<chunksize> > wl2ty;
  //typedef GaloisRuntime::WorkList::ChunkedFIFO<chunksize, ND*, true> wl2ty;
  galois::InsertBag<ND*> wl2;
#else
//  const int chunksize = 64;
//  std::cerr << "Using chunk size : " << chunksize << std::endl;
//  typedef GaloisRuntime::WorkList::SimpleOrderedByIntegerMetric<EdgeIndexer, GaloisRuntime::WorkList::dChunkedLIFO<chunksize, ED*, true>, true, ED*, true> wl2ty;
//  GaloisRuntime::galois_insert_bag<ED*> wl2;
#endif
  //GaloisRuntime::WorkList::OrderedByIntegerMetric<ED*, EdgeIndexer, GaloisRuntime::WorkList::dChunkedFIFO<ED*, 64, true>, true> wl2;
  typedef galois::WorkList::dChunkedLIFO<16> wl4ty;  
  //typedef GaloisRuntime::WorkList::dChunkedLIFO<16, ND*,  true> wl4ty;  
  //typedef GaloisRuntime::WorkList::OrderedByIntegerMetric<f2ItemIndexer, GaloisRuntime::WorkList::dChunkedLIFO<32, f2Item, true>, true, f2Item, true> wl4ty;
  galois::InsertBag<ND*> wl4;
	fringewl = &wl4;
//  GaloisRuntime::WorkList::FIFO<int,  true> wl5;
#else
  galois::WorkList::GFIFO<ED*, false> wl2;
  galois::WorkList::GFIFO<ND*, false> wl4;
#endif

//  for (int kk=0; kk<numThreads; ++kk) {
//    std::vector<f2Item> & abuff = fringeBuffs.get(kk);
//    abuff.reserve(nnodes);
//  }
//  galois::Statistic("Mem1", GaloisRuntime::MM::pageAllocInfo());
//  galois::preAlloc(6000);
//  galois::Statistic("Mem2", GaloisRuntime::MM::pageAllocInfo());
	//cerr << "OMP Threads: " << omp_get_num_threads() << endl;
  galois::StatTimer T;
  T.start();
  totalTimer.start();
	for (int i=startNode; i<nnodes; ++i) {
//    galois::Statistic("Mem3", GaloisRuntime::MM::pageAllocInfo());
  
		ND * active = &(gnodes[i].data);
    currSrcNode = active;
		int nnbrs = graph->outNeighborsSize(active);
		if (nnbrs == 0) {
			continue;
		}
    
    stepCnt++;
   if (stepCnt >= 2) break;

#if MERGE_LOOPS		
	 curr_round++;
#endif

		//std::list<ED*> wl;
#if USE_NODE_BASED
         std::vector<ND*>  wl;
    wl2.push_back(active);
#else
		std::vector<ED*> wl;
		graph->initWLToOutEdges(active, wl2);
#endif
//		wl2.fill_initial(wl.begin(), wl.end());
    active->initAsSource();
    { cerr << "Source is " << active->toString() << endl; } 
    if (DBG) { cerr << "Source is " << active->toString() << endl; } 
    //graph->checkClearEdges();
//__itt_resume();
    firstLoopTimer.start();
#if USE_NODE_BASED

#if CONCURRENT
    galois::for_each(wl2.begin(), wl2.end(), feach1NodeBased, galois::wl<wl2ty>());
    wl2.clear();
//		galois::for_each<wl2ty>(active, feach1NodeBased);
#else
	Timer tt;
		tt.start();

	while (!wl2.empty()) {
      Nd *nn = wl2.pop().second;
      feach1NodeBased(nn, wl2);
    }
	tt.stop();
	cerr << "tt " << tt.get() << endl;

#endif

#else

#if CONCURRENT
    galois::Timer firstLT;
    firstLT.start();
    galois::for_each(wl2.begin(), wl2.end(), feach1, galois::wl<wl2ty>());
    firstLT.stop();
    std::cerr << "FLTime: " << firstLT.get() << std::endl;
#else
	Timer tt;
		tt.start();

	while (!wl2.empty()) {
      ED *ee = wl2.pop().second;
      feach1(ee, wl2);
    }
	tt.stop();
	cerr << "tt " << tt.get() << endl;

#endif

#endif
    firstLoopTimer.stop();
//__itt_pause();
		if (DOCHECKS) graph->checkGraph(active);

    secondLoopTimer.start();
#if DBG_FRINGECNT
		int fringeCnt = 0;
#endif
#if CONCURRENT

#if DBG_FRINGECNT
	for (int kk=0; kk<numThreads; ++kk) {
		fringeCnts.get(kk) = 0;
	}
#endif
	//GaloisRuntime::for_all_parallel(findFringe);
//  fringeFindOMP();
//  galois::on_each(findFringe);
//__itt_resume();
  galois::do_all(boost::counting_iterator<int>(0), boost::counting_iterator<int>(nnodes), findFringe);
//  __itt_pause();
//	for (int kk=0; kk<numThreads; ++kk) {
//    std::vector<f2Item> & abuff = fringeBuffs.get(kk);
//    std::vector<f2Item>::const_iterator it = abuff.begin();
//    std::vector<f2Item>::const_iterator itend = abuff.end();
//    while (it != itend) {
//		  fringewl->push_back(*it);
//      it++;
//    }
//    std::cerr << "buff " << kk << " size " << abuff.size() << "\n";
//    abuff.clear();
//  }
#if DBG_FRINGECNT
	for (int kk=0; kk<numThreads; ++kk) {
		int tmp = fringeCnts.get(kk);
		fringeCnts.get(kk) = 0;
		fringeCnt += tmp;
	}
#endif

#else
		//std::list<ND*> wl3;
		for (int j=0; j<nnodes; ++j) {
			ND * n = &(gnodes[j].data);
			if (n->nsuccs == 0) {
#if DBG_FRINGECNT
				fringeCnt++;
#endif
				wl4.push_back(n);
			}
		}
		//wl4.fill_initial(wl3.begin(), wl3.end());
#endif
		secondLoopTimer.stop();
#if DBG_FRINGECNT
		cerr << fringeCnt << " nodes in fringe " << endl;
#endif
		thirdLoopTimer.start();
		double backupSrcBC = currSrcNode->bc;
#if CONCURRENT	 
                galois::for_each_local(wl4, feach2, galois::wl<wl4ty>());
#else
    while (!wl4.empty()) {
      ND *nn = wl4.pop().second;
      feach2(nn, wl4);
    }
#endif
		currSrcNode->bc = backupSrcBC;
    thirdLoopTimer.stop();
    //std::cerr << "Is wl empty ? " << wl4.empty() << std::endl;
    wl4.clear();
    //std::cerr << "Is wl empty ? " << wl4.empty() << std::endl;
    if (DOCHECKS) graph->checkSteadyState2();
    //graph->printGraph();
    fourthLoopTimer.start();
#if CONCURRENT
// 		wl5.fill_initial(workChunks.begin(), workChunks.end());
//    galois::for_each(wl5, feach4);
#if MERGE_LOOPS
#else
//    galois::on_each(cleanupGloop);
	//	GaloisRuntime::for_all_parallel(cleanupGloop);
#endif
		//graph->cleanupDataOMP();
#else
    graph->cleanupData();
#endif
    fourthLoopTimer.stop();
	}
  totalTimer.stop();
  T.stop();
//  galois::Statistic("Mem4", GaloisRuntime::MM::pageAllocInfo());
  std::cout << "Total Time " << totalTimer.get() << std::endl;
  std::cout<< "First Loop: " << firstLoopTimer.get() << std::endl;
  std::cout<< "Second Loop: " << secondLoopTimer.get() << std::endl;
  std::cout<< "Third Loop: " << thirdLoopTimer.get() << std::endl;
  std::cout<< "Fourth Loop: " << fourthLoopTimer.get() << std::endl;
 
#ifdef COUNT_ACTIONS
  unsigned long sum1 = 0, sum2 = 0, sum2ex = 0, sum3 = 0, sum4 = 0, sum555 = 0, sumNone = 0;
  for (int i=0; i<numThreads; ++i) {
  	sum1 += action1cnt.get(i);
	sum2 += action2cnt.get(i);
	sum2ex += action2Excnt.get(i);
	sum3 += action3cnt.get(i);
	sum4 += action4cnt.get(i);
	sum555 += action555.get(i);
  sumNone += actionNone.get(i);
  }
  std::cout << "Action 1 " << sum1 << "\nAction 2 " << sum2  << "\nRealActionUS " << sum2ex << "\nAction 3 " << sum3 << "\nAction 4 " << sum4 << "\nAction4_mut " << sum555 << "\nActionNone " << sumNone << endl;
//  for (int i=0; i<numThreads; ++i) 
//  	std::cout << "Largest Node distance is " << largestNodeDist.get(i) << endl;
	
#endif

  //if (!skipVerify)
  //  graph->verify();
  if (!skipVerify) {
    int count = 0;
    for (int i = 0; i << nnodes && count < 10; ++i, ++count) {
      std::cout << count << ": "
        << std::setiosflags(std::ios::fixed) << std::setprecision(6)
        << gnodes[i].data.bc
        << "\n";
    }
  }

  graph->printBCs();
#if GENERATE_CERTIFICATE
  graph->printAllBCs(numThreads, "certificate_");
#endif
  return 0;
}
