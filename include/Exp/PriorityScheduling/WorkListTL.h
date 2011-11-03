/**
 * Common support for worklist experiments.
 */
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/PaddedLock.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/Threads.h"
#include <boost/utility.hpp>
#include <tbb/concurrent_hash_map.h>

#include <iosfwd>

namespace Exp {
static std::string WorklistName;


template<class Indexer, typename ContainerTy = GaloisRuntime::WorkList::TbbFIFO<>, typename T = int, bool concurrent = true >
class SimpleOrderedByIntegerMetric : private boost::noncopyable, private GaloisRuntime::PaddedLock<concurrent> {

  using GaloisRuntime::PaddedLock<concurrent>::lock;
  using GaloisRuntime::PaddedLock<concurrent>::try_lock;
  using GaloisRuntime::PaddedLock<concurrent>::unlock;

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
  GaloisRuntime::PerCPU<perItem> current;

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

template<class Indexer, typename ContainerTy, bool concurrent = true, int binmax = 1024*1024 >
class BarrierOBIM : private boost::noncopyable {
  typedef typename ContainerTy::template rethread<concurrent>::WL CTy;

  volatile unsigned int current;
  volatile unsigned int pushmax;

  CTy* B;

  Indexer I;

  GaloisRuntime::TerminationDetection term;
  pthread_barrier_t barr1;
  pthread_barrier_t barr2;

 public:
  template<bool newconcurrent>
  struct rethread {
    typedef BarrierOBIM<Indexer,ContainerTy,newconcurrent,binmax> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef BarrierOBIM<Indexer,typename ContainerTy::template retype<Tnew>::WL,concurrent, binmax> WL;
  };

  typedef typename CTy::value_type value_type;

  BarrierOBIM(const Indexer& x = Indexer())
    :current(0), pushmax(0), I(x)
  {
    B = new CTy[binmax];
    //std::cerr << "$"<<getSystemThreadPool().getActiveThreads() <<"$";
    pthread_barrier_init(&barr1, NULL, GaloisRuntime::getSystemThreadPool().getActiveThreads());
    pthread_barrier_init(&barr2, NULL, GaloisRuntime::getSystemThreadPool().getActiveThreads());
  }
  ~BarrierOBIM() {
    delete[] B;
  }

  bool push(value_type val) {
    unsigned int index = I(val);
    //std::cerr << "P: " << index << "\n";
    if (index < current)
      index = current;
    if (index >= binmax)
      index = binmax - 1;
    term.workHappened();
    unsigned int oldi;
    while (index > (oldi = pushmax)) {
      __sync_bool_compare_and_swap(&pushmax, oldi, index);
    }
    return B[index].push(val);
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
    return true;
  }

  std::pair<bool, value_type> pop() {
    do {
      if (current > pushmax)
	return std::make_pair(false, value_type());
      do {
	//Find a successful pop
	std::pair<bool, value_type> retval = B[current].pop();
	if (retval.first) {
	  term.workHappened();
	  return retval;
	}
	term.localTermination();
      } while (!term.globalTermination());
      
      pthread_barrier_wait(&barr1);
  
      if (GaloisRuntime::ThreadPool::getMyID() == 0) {
	//std::cerr << "inc: " << current << "\n";
	term.reset();
	if (current <= pushmax)
	  __sync_fetch_and_add(&current, 1);
      }

      pthread_barrier_wait(&barr2);
    } while (true);
  }
};


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


template<typename DefaultWorklist,typename dChunk,typename Chunk,typename Indexer,typename Less,typename Greater>
struct StartWorklistExperiment {
  template<typename Iterator,typename Functor>
  void operator()(std::ostream& out, Iterator ii, Iterator ei, Functor fn, std::string wlname = "") {
    using namespace GaloisRuntime::WorkList;

    typedef OrderedByIntegerMetric<Indexer, dChunk> OBIM;
    typedef TbbPriQueue<Greater> TBB;
    typedef LocalStealing<TBB> LTBB;
    typedef SkipListQueue<Less> SLQ;
    typedef SimpleOrderedByIntegerMetric<Indexer> SOBIM;
    typedef LocalStealing<SOBIM> LSOBIM;
    typedef OrderedByIntegerMetric<Indexer, Chunk> NAOBIM;
    typedef BarrierOBIM<Indexer, dChunk> BOBIM;
    typedef CTOrderedByIntegerMetric<Indexer, dChunk> CTOBIM;
    typedef CTOrderedByIntegerMetric<Indexer, Chunk> NACTOBIM;

    typedef WorkListTracker<Indexer, OBIM> TR_OBIM;
    typedef WorkListTracker<Indexer, TBB>  TR_TBB;
    typedef WorkListTracker<Indexer, LTBB> TR_LTBB;
    typedef WorkListTracker<Indexer, SLQ>  TR_SLQ;
    typedef WorkListTracker<Indexer, SOBIM> TR_SOBIM;
    typedef WorkListTracker<Indexer, LSOBIM> TR_LSOBIM;
    typedef WorkListTracker<Indexer, NAOBIM> TR_NAOBIM;
    typedef WorkListTracker<Indexer, CTOBIM> TR_CTOBIM;
    typedef WorkListTracker<Indexer, NACTOBIM> TR_NACTOBIM;
    typedef WorkListTracker<Indexer, BOBIM> TR_BOBIM;

    typedef NoInlineFilter<OBIM> NI_OBIM;
    typedef NoInlineFilter<TBB>  NI_TBB;
    typedef NoInlineFilter<LTBB> NI_LTBB;
    typedef NoInlineFilter<SLQ>  NI_SLQ;
    typedef NoInlineFilter<SOBIM> NI_SOBIM;
    typedef NoInlineFilter<LSOBIM> NI_LSOBIM;
    typedef NoInlineFilter<NAOBIM> NI_NAOBIM;
    typedef NoInlineFilter<CTOBIM> NI_CTOBIM;
    typedef NoInlineFilter<NACTOBIM> NI_NACTOBIM;
    typedef NoInlineFilter<BOBIM> NI_BOBIM;

    std::string name = (wlname == "") ? WorklistName : wlname;
#define WLFOO(__x, __y) else if (name == #__x) {\
  out << "Using worklist: " << name << "\n"; \
  Galois::for_each<__y >(ii, ei, fn); } 

    if (name == "default" || name == "") {
      out << "Using worklist: default\n";
      Galois::for_each<DefaultWorklist>(ii, ei, fn); 
    }
    WLFOO(obim, OBIM)
    WLFOO(sobim, SOBIM)
    WLFOO(lsobim, LSOBIM)
    WLFOO(naobim, NAOBIM)
    WLFOO(ctobim, CTOBIM)
    WLFOO(nactobim, NACTOBIM)
    WLFOO(slq, SLQ)
    WLFOO(tbb, TBB)
    WLFOO(ltbb, LTBB)
    WLFOO(bobim, BOBIM)
    WLFOO(tr-obim, TR_OBIM)
    else {
      out << "Unrecognized worklist " << name << "\n";
    }
#undef WLFOO
  }
};

} // end namespace
