/**
 * Common support for worklist experiments.
 */
#ifndef PRIORITYSCHEDULING_WORKLIST_H
#define PRIORITYSCHEDULING_WORKLIST_H

#include "Galois/WorkList/WorkList.h"
#include "Galois/WorkList/WorkListExperimental.h"
#include "Galois/Runtime/ll/gio.h"

namespace Exp {

__attribute__((weak)) llvm::cl::opt<std::string> WorklistName("wl", llvm::cl::desc("Worklist to use"), llvm::cl::init("DEFAULT"));

using namespace Galois::Runtime::LL;
using namespace Galois::WorkList;

template<int CS, bool LF>
struct PickInner;

template<int CS>
struct PickInner<CS, true> {
  typedef dChunkedLIFO<CS> dChunk;
  typedef  ChunkedLIFO<CS> Chunk;
};
template<int CS>
struct PickInner<CS, false> {
  typedef dChunkedFIFO<CS> dChunk;
  typedef  ChunkedFIFO<CS> Chunk;
};

 template<int ChunkSize, typename Ind, typename DEFAULT, typename Less, typename Greater, bool LF = false >
struct PriAuto {

   typedef typename PickInner<ChunkSize, LF>::dChunk dChunk;
   typedef typename PickInner<ChunkSize, LF>::Chunk   Chunk;

  //OBIM
  typedef   OrderedByIntegerMetric<Ind,dChunk, true> OBIM_DMB;
  typedef CTOrderedByIntegerMetric<Ind,dChunk, true> OBIM_DSB;
  typedef   OrderedByIntegerMetric<Ind, Chunk, true> OBIM_CMB;
  typedef CTOrderedByIntegerMetric<Ind, Chunk, true> OBIM_CSB;
  typedef   OrderedByIntegerMetric<Ind,dChunk,false> OBIM_DMN;
  typedef CTOrderedByIntegerMetric<Ind,dChunk,false> OBIM_DSN;
  typedef   OrderedByIntegerMetric<Ind, Chunk,false> OBIM_CMN;
  typedef CTOrderedByIntegerMetric<Ind, Chunk,false> OBIM_CSN;

  //TBB
#ifdef USE_TBB
  typedef TbbPriQueue<Greater> TBB;
  typedef PTbb<Greater> PTBB;
  typedef STbb<Greater> STBB;
#endif
  //MISC
  typedef SkipListQueue<Less> SLQ;
  typedef SetQueue<Less> SETQ;



  template<typename IterTy,typename FunctionTy>
  static void for_each(IterTy b, IterTy e, FunctionTy f, const char* loopname = 0) {
    static bool printed = false;
#define WLFOO2(__x)							\
    if (WorklistName == #__x) {						\
      if (!printed) {							\
	gInfo("WorkList %s\n", #__x);					\
	printed = true;							\
      }									\
      Galois::for_each<__x>(b,e,f,loopname);				\
    } else
#include "PrioritySchedulers.h"
#undef WLFOO2
#define WLFOO2(__x)							\
    if (WorklistName == "NI_" #__x) {					\
      if (!printed) {							\
	gInfo("WorkList %s\n", "NI_" #__x);				\
	printed = true;							\
      }									\
      Galois::for_each<NoInlineFilter<__x> >(b,e,f,loopname);		\
    } else
#include "PrioritySchedulers.h"
#undef WLFOO2

    {
      GALOIS_ERROR(true, "unknown Worklist [%s]", WorklistName.c_str());
    }
  }
  template<typename InitItemTy,typename FunctionTy>
  static void for_each(InitItemTy i, FunctionTy f, const char* loopname = 0) {
    InitItemTy wl[1] = {i};
    for_each(&wl[0], &wl[1], f, loopname);
  }
};

} // end namespace

#endif
