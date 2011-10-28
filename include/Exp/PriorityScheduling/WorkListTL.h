/**
 * Common support for worklist experiments.
 */
#include "Galois/Runtime/WorkList.h"
#include <tbb/concurrent_hash_map.h>

#include <iosfwd>

namespace Exp {
static std::string WorklistName;


template<class Indexer, typename ContainerTy = GaloisRuntime::WorkList::TbbFIFO<>, typename T = int, bool concurrent = true >
class SimpleOrderedByIntegerMetric : private boost::noncopyable, private PaddedLock<concurrent> {

  using PaddedLock<concurrent>::lock;
  using PaddedLock<concurrent>::try_lock;
  using PaddedLock<concurrent>::unlock;

  typedef ContainerTy CTy;

  CTy* current;
  int cint;
  std::map<int, CTy*> mapping;
  Indexer I;

  CTy* updateLocalOrCreate(int i) {
    //Try local then try update then find again or else create and update the master log
    //check if current bin is the right thing to use (lock-free)
    if (i == cint) {
      CTy* n = current;
      if (n) return n;
    }
    lock();
    CTy*& lC = mapping[i];
    if (!lC)
      lC = new CTy();
    unlock();
    return lC;
  }

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef  SimpleOrderedByIntegerMetric<Indexer,ContainerTy,T,newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef SimpleOrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>::WL,Tnew,concurrent> WL;
  };
  
  typedef T value_type;
  
  SimpleOrderedByIntegerMetric(const Indexer& x = Indexer())
    :current(0), I(x)
  { }

  bool push(value_type val) {
    unsigned int index = I(val);
    CTy* lC = updateLocalOrCreate(index);
    bool retval = lC->push(val);
    return retval;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
    return true;
  }

  std::pair<bool, value_type> pop() {
    //Fastpath
    CTy* c = current;
    std::pair<bool, value_type> retval;
    if (c && (retval = c->pop()).first)
      return retval;

    //Failed, find minimum bin
    retval.first = false;
    //    if (ThreadPool::getMyID() == 1) {
      lock();
    if (current != c) {
      unlock();
      return pop();
    }
      for (typename std::map<int, CTy*>::iterator ii = mapping.begin(), ee = mapping.end(); ii != ee; ++ii) {
	current = ii->second;
	if ((retval = current->pop()).first) {
	  cint = ii->first;
	  goto exit;
	}
      }
      retval.first = false;
    exit:
      unlock();
    //   }
    return retval;
  }
};

template<class Indexer, typename ContainerTy = GaloisRuntime::WorkList::ChunkedLIFO<16>, typename T = int, bool concurrent = true >
class CTOrderedByIntegerMetric : private boost::noncopyable {

  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  struct perItem {
    CTy* current;
    unsigned int curVersion;
  };

  typedef tbb::concurrent_hash_map<int, CTy*> HM;
  HM wl;
  int maxV;

  Indexer I;
  PerCPU<perItem> current;

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef  CTOrderedByIntegerMetric<Indexer,ContainerTy,T,newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef CTOrderedByIntegerMetric<Indexer,typename ContainerTy::template retype<Tnew>::WL,Tnew,concurrent> WL;
  };

  typedef T value_type;

 CTOrderedByIntegerMetric(const Indexer& x = Indexer())
   :maxV(1),I(x)
  {
    for (unsigned int i = 0; i < current.size(); ++i) {
      current.get(i).current = 0;
    }
  }

  bool push(value_type val) {
    unsigned int index = I(val);
    perItem& pI = current.get();
    //fastpath
    if (index == pI.curVersion && pI.current)
      return pI.current->push(val);
    //slow path
    bool retval;
    if (wl.count(index)) {
      typename HM::const_accessor a;
      wl.find(a, index);
      retval = a->second->push(val);
    } else {
      typename HM::accessor a;
      wl.insert(a, index);
      if (!a->second)
	a->second = new CTy();
      retval = a->second->push(val);
    }
    unsigned int oldMax;
    while ((oldMax = maxV) < index)
      __sync_bool_compare_and_swap(&maxV, oldMax, index);
    return retval;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
    return true;
  }

  std::pair<bool, value_type> pop() {
    //Find a successful pop
    perItem& pI = current.get();
    CTy*& C = pI.current;
    std::pair<bool, value_type> retval;
    if (C && (retval = C->pop()).first)
      return retval;
    //Failed, find minimum bin
    for (int i = 0; i <= maxV; ++i) {
      typename HM::const_accessor a;
      if (wl.find(a, i)) {
	pI.curVersion = i;
	C = a->second;
	if (C && (retval = C->pop()).first)
	  return retval;
      }
    }
    retval.first = false;
    return retval;
  }
};
WLCOMPILECHECK(CTOrderedByIntegerMetric);

static void parse_worklist_command_line(std::vector<const char*>& args) {
  for (std::vector<const char*>::iterator ii = args.begin(), ei = args.end(); ii != ei; ++ii) {
    if (strcmp(*ii, "-wl") == 0 && ii + 1 != ei) {
      WorklistName = ii[1];
      ii = args.erase(ii);
      ii = args.erase(ii);
      --ii;
      ei = args.end();
    }
  }
}

//! there is std::not1() but worklists take types not instances :(
template<typename Functor>
struct NotLess: public std::binary_function<
                typename Functor::first_argument_type,
                typename Functor::second_argument_type,
                bool> {
    Functor m_fn;
  
  bool operator()(
      typename Functor::first_argument_type x,
      typename Functor::second_argument_type y) const {
    // XXX(ddn): != is not right because it will usually be between GraphNodes...
    return !m_fn(x, y) && x != y;
  }
};

template<typename DefaultWorklist,typename Indexer,typename Less>
struct StartWorklistExperiment {
  template<typename Iterator,typename Functor>
  void operator()(std::ostream& out, Iterator ii, Iterator ei, Functor fn) {
    using namespace GaloisRuntime::WorkList;
    typedef dChunkedLIFO<16> IChunk;
    typedef ChunkedLIFO<16> NAChunk;

    typedef OrderedByIntegerMetric<Indexer, IChunk> OBIM;
    typedef TbbPriQueue<NotLess<Less> > TBB;
    typedef LocalStealing<TBB> LTBB;
    typedef SkipListQueue<Less> SLQ;
    typedef SimpleOrderedByIntegerMetric<Indexer> SOBIM;
    typedef LocalStealing<SOBIM> LSOBIM;
    typedef OrderedByIntegerMetric<Indexer, NAChunk> NAOBIM;
    typedef CTOrderedByIntegerMetric<Indexer, IChunk> CTOBIM;

    typedef WorkListTracker<Indexer, OBIM> TR_OBIM;
    typedef WorkListTracker<Indexer, TBB>  TR_TBB;
    typedef WorkListTracker<Indexer, LTBB> TR_LTBB;
    typedef WorkListTracker<Indexer, SLQ>  TR_SLQ;
    typedef WorkListTracker<Indexer, SOBIM> TR_SOBIM;
    typedef WorkListTracker<Indexer, LSOBIM> TR_LSOBIM;
    typedef WorkListTracker<Indexer, NAOBIM> TR_NAOBIM;
    typedef WorkListTracker<Indexer, CTOBIM> TR_CTOBIM;

    typedef NoInlineFilter<OBIM> NI_OBIM;
    typedef NoInlineFilter<TBB>  NI_TBB;
    typedef NoInlineFilter<LTBB> NI_LTBB;
    typedef NoInlineFilter<SLQ>  NI_SLQ;
    typedef NoInlineFilter<SOBIM> NI_SOBIM;
    typedef NoInlineFilter<LSOBIM> NI_LSOBIM;
    typedef NoInlineFilter<NAOBIM> NI_NAOBIM;
    typedef NoInlineFilter<CTOBIM> NI_CTOBIM;

#define WLFOO(__x, __y) else if (WorklistName == #__x) {\
  out << "Using worklist: " << WorklistName << "\n"; \
  Galois::for_each<__y >(ii, ei, fn); } 

    if (WorklistName == "default" || WorklistName == "") {
      out << "Using worklist: default\n";
      Galois::for_each<DefaultWorklist>(ii, ei, fn); 
    }
    WLFOO(obim, OBIM)
    WLFOO(sobim, SOBIM)
    WLFOO(lsobim, LSOBIM)
    WLFOO(naobim, NAOBIM)
    WLFOO(ctobim, CTOBIM)
    WLFOO(slq, SLQ)
    WLFOO(tbb, TBB)
    WLFOO(ltbb, LTBB)
    WLFOO(chunkedfifo, dChunkedFIFO<16> )
    WLFOO(fifo, FIFO<> )
    WLFOO(lifo, LIFO<> )
    else {
      out << "Unrecognized worklist " << WorklistName << "\n";
    }
#undef WLFOO
  }
};

} // end namespace
