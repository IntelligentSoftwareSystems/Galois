/**
 * Common support for worklist experiments.
 */
#ifndef PRIORITYSCHEDULING_WORKLIST_H
#define PRIORITYSCHEDULING_WORKLIST_H

#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/WorkListExperimental.h"
#include "Galois/Runtime/ll/gio.h"

namespace Exp {

__attribute__((weak)) llvm::cl::opt<std::string> WorklistName("wl", llvm::cl::desc("Worklist to use"), llvm::cl::init("DEFAULT"));

using namespace GaloisRuntime::LL;
using namespace GaloisRuntime::WorkList;

template<int ChunkSize, typename Ind, typename DEFAULT, typename Less, typename Greater >
struct PriAuto {

  typedef dChunkedLIFO<ChunkSize> dChunk;
  typedef  ChunkedLIFO<ChunkSize>  Chunk;

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
  typedef TbbPriQueue<Greater> TBB;
  typedef PTbb<Greater> PTBB;
  typedef STbb<Greater> STBB;

  //MISC
  typedef SkipListQueue<Less> SLQ;
  typedef SetQueue<Less> SETQ;



  template<typename IterTy,typename FunctionTy>
  static void for_each(IterTy b, IterTy e, FunctionTy f, const char* loopname = 0) {

#define WLFOO2(__x)							\
    if (WorklistName == #__x) {						\
      gInfo("WorkList %s\n", #__x);					\
      Galois::for_each<__x>(b,e,f,loopname);				\
    } else
#include "PrioritySchedulers.h"
#undef WLFOO2
#define WLFOO2(__x)							\
    if (WorklistName == "NI_" #__x) {					\
      gInfo("WorkList %s\n", "NI_" #__x);				\
      Galois::for_each<NoInlineFilter<__x> >(b,e,f,loopname);		\
    } else
#include "PrioritySchedulers.h"
#undef WLFOO2

    {
      gError(true, "Unknown Worklist [%s]\n", WorklistName.c_str());
    }
  }
};

} // end namespace

#endif
