/**
 * Common support for worklist experiments.
 */
#ifndef PRIORITYSCHEDULING_WORKLIST_H
#define PRIORITYSCHEDULING_WORKLIST_H

#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/WorkListExperimental.h"

// Uncomment to enable bunches of worklist types, disabled to speed up builds
//#define _RUN_EXP

namespace Exp {
__attribute__((weak)) llvm::cl::opt<std::string> WorklistName("wl", llvm::cl::desc("Worklist to use"));

template<
 typename DefaultWorklist,
 typename dChunk,
 typename Chunk,
 typename Indexer,
 typename Less,
 typename Greater
 >
struct WorklistExperiment {
#ifndef _RUN_EXP
  template<typename Iterator,typename Functor>
  void for_each(std::ostream& out, Iterator ii, Iterator ei, Functor fn) {
    Galois::for_each<DefaultWorklist>(ii, ei, fn); 
  }
#else
  template<typename Iterator,typename Functor>
  void for_each(std::ostream& out, Iterator ii, Iterator ei, Functor fn) {
    using namespace GaloisRuntime::WorkList;

    typedef OrderedByIntegerMetric<Indexer, dChunk> OBIM;
#ifdef GALOIS_USE_TBB
    typedef TbbPriQueue<Greater> TBB;
    typedef LocalStealing<TBB> LTBB;
    typedef PTbb<Greater> PTBB;
    typedef CTOrderedByIntegerMetric<Indexer, dChunk> CTOBIM;
    typedef CTOrderedByIntegerMetric<Indexer, Chunk> NACTOBIM;
#endif
    typedef ChunkedFIFO<256> CF256;
    typedef ChunkedLIFO<256> CL256;
    typedef dChunkedFIFO<256> DCF256;
    typedef dChunkedLIFO<256> DCL256;
    typedef SkipListQueue<Less> SLQ;
    typedef SimpleOrderedByIntegerMetric<Indexer> SOBIM;
    typedef LocalStealing<SOBIM> LSOBIM;
    typedef OrderedByIntegerMetric<Indexer, Chunk> NAOBIM;
    typedef BarrierOBIM<Indexer, dChunk> BOBIM;
    typedef LevelStealing<Random<> > RANDOM;
    typedef StaticPartitioning<> STATIC;

    std::string name = WorklistName;

#define WLFOO(__b, __e, __p, __x, __y)					\
  if (name == #__x) {							\
    out << "Using worklist: " << name << "\n";				\
    Galois::for_each<__y>(__b, __e, __p);				\
  } else if (name == "tr_" #__x) {					\
    out << "Using worklist: " << name << "\n";			\
    Galois::for_each<WorkListTracker<Indexer, __y > >(__b, __e, __p);	\
  } else if (name == "ni_" #__x) {					\
    out << "Using worklist: " << name << "\n";			\
    Galois::for_each<NoInlineFilter< __y > >(__b, __e, __p);		\
  }

    if (name == "default" || name == "") {
      out << "Using worklist: default\n";
      Galois::for_each<DefaultWorklist>(ii, ei, fn); 
    } else
    WLFOO(ii, ei, fn, lifo,     LIFO<>)   else
    WLFOO(ii, ei, fn, fifo,     FIFO<>)   else
    WLFOO(ii, ei, fn, obim,     OBIM)     else
    WLFOO(ii, ei, fn, sobim,    SOBIM)    else
    WLFOO(ii, ei, fn, lsobim,   LSOBIM)   else
    WLFOO(ii, ei, fn, naobim,   NAOBIM)   else
    WLFOO(ii, ei, fn, slq,      SLQ)      else
    WLFOO(ii, ei, fn, bobim,    BOBIM)    else
    WLFOO(ii, ei, fn, dchunk,   dChunk)   else
    WLFOO(ii, ei, fn, chunk,    Chunk)    else
    WLFOO(ii, ei, fn, chunkfifo, CF256)   else
    WLFOO(ii, ei, fn, chunklifo, CL256)   else
    WLFOO(ii, ei, fn, dchunkfifo, DCF256) else
    WLFOO(ii, ei, fn, dchunklifo, DCL256) else
    WLFOO(ii, ei, fn, random,   RANDOM)   else
    WLFOO(ii, ei, fn, sp,       STATIC)   else
#ifdef GALOIS_USE_TBB
    WLFOO(ii, ei, fn, nactobim, NACTOBIM) else
    WLFOO(ii, ei, fn, ctobim,   CTOBIM)   else
    WLFOO(ii, ei, fn, tbb,      TBB)      else
    WLFOO(ii, ei, fn, ltbb,     LTBB)     else
    WLFOO(ii, ei, fn, ptbb,     PTBB)     else
#endif
    {
      out << "Unrecognized worklist " << name << "\n";
    }
#undef WLFOO
  }
#endif
};

} // end namespace

#endif
