/**
 * Common support for worklist experiments.
 */
#include "Galois/Runtime/WorkList.h"
#include "Galois/Runtime/ll/PaddedLock.h"
#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/Threads.h"
#include <boost/utility.hpp>
#include <tbb/concurrent_hash_map.h>
#include "Galois/Runtime/DebugWorkList.h"

#ifdef GALOIS_TBB
#define TBB_PREVIEW_CONCURRENT_PRIORITY_QUEUE 1
#include <tbb/concurrent_priority_queue.h>
#include <tbb/concurrent_queue.h>
#endif


#include <iosfwd>

namespace Exp {
  __attribute__((weak)) llvm::cl::opt<std::string> WorklistName("wl", llvm::cl::desc("Worklist to use"));

#ifdef GALOIS_TBB
template<typename T = int>
class TbbFIFO : private boost::noncopyable  {
  tbb::concurrent_bounded_queue<T> wl;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef TbbFIFO<T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef TbbFIFO<Tnew> WL;
  };

  typedef T value_type;

  bool push(value_type val) {
    wl.push(val);
    return true;
  }

  bool pushi(value_type val) {
    return push(val);
  }

  boost::optional<value_type> pop() {
    T V = T();
    bool B = wl.try_pop(V);
    if (B)
      return boost::optional<value_type>(V);
    return boost::optional<value_type>();
  }
};
WLCOMPILECHECK(TbbFIFO);

#endif // GALOIS_TBB


 template<class Indexer, typename ContainerTy = GaloisRuntime::WorkList::FIFO<>, typename T = int, bool concurrent = true >
   class SimpleOrderedByIntegerMetric : private boost::noncopyable, private GaloisRuntime::LL::PaddedLock<concurrent> {

   using GaloisRuntime::LL::PaddedLock<concurrent>::lock;
   using GaloisRuntime::LL::PaddedLock<concurrent>::try_lock;
   using GaloisRuntime::LL::PaddedLock<concurrent>::unlock;

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

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }
   
  boost::optional<value_type> pop() {
    //Fastpath
    CTy* c = current;
    boost::optional<value_type> retval;
    if (c && (retval = c->pop()))
      return retval;
    
    //Failed, find minimum bin
    //    if (ThreadPool::getMyID() == 1) {
    lock();
    if (current != c) {
      unlock();
      return pop();
    }
    for (typename std::map<int, CTy*>::iterator ii = mapping.begin(), ee = mapping.end(); ii != ee; ++ii) {
      current = ii->second;
      if ((retval = current->pop())) {
	cint = ii->first;
	goto exit;
      }
    }
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

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop() {
    //Find a successful pop
    perItem& pI = current.get();
    CTy*& C = pI.current;
    boost::optional<value_type> retval;
    if (C && (retval = C->pop()))
      return retval;
    //Failed, find minimum bin
    for (int i = 0; i <= maxV; ++i) {
      typename HM::const_accessor a;
      if (wl.find(a, i)) {
	pI.curVersion = i;
	C = a->second;
	if (C && (retval = C->pop()))
	  return retval;
      }
    }
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
  
  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop() {
    do {
      if (current > pushmax)
	return boost::optional<value_type>();
      do {
	//Find a successful pop
	boost::optional<value_type> retval = B[current].pop();
	if (retval) {
	  term.workHappened();
	  return retval;
	}
	term.localTermination();
      } while (!term.globalTermination());
      
      pthread_barrier_wait(&barr1);
  
      if (GaloisRuntime::LL::getTID() == 0) {
	//std::cerr << "inc: " << current << "\n";
	term.reset();
	if (current <= pushmax)
	  __sync_fetch_and_add(&current, 1);
      }

      pthread_barrier_wait(&barr2);
    } while (true);
  }
};

template<typename T = int, bool concurrent = true>
  class Random : private boost::noncopyable, private GaloisRuntime::LL::PaddedLock<concurrent>  {
  std::vector<T> wl;
  unsigned int seed;
  using GaloisRuntime::LL::PaddedLock<concurrent>::lock;
  using GaloisRuntime::LL::PaddedLock<concurrent>::try_lock;
  using GaloisRuntime::LL::PaddedLock<concurrent>::unlock;

  unsigned int nextRand() {
    seed = 214013*seed + 2531011; 
    return (seed >> 16) & 0x7FFF;
  }

public:
  Random(): seed(0xDEADBEEF) { }

  template<bool newconcurrent>
  struct rethread {
    typedef Random<T, newconcurrent> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef Random<Tnew, concurrent> WL;
  };

  typedef T value_type;

  bool push(value_type val) {
    lock();
    wl.push_back(val);
    unlock();
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    lock();
    while (b != e)
      wl.push_back(*b++);
    unlock();
    return true;
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop() {
    lock();
    if (wl.empty()) {
      unlock();
      return boost::optional<value_type>();
    } else {
      size_t size = wl.size();
      unsigned int index = nextRand() % size;
      value_type retval = wl[index];
      std::swap(wl[index], wl[size-1]);
      wl.pop_back();
      unlock();
      return boost::optional<value_type>(retval);
    }
  }
};
WLCOMPILECHECK(Random);


template <typename T> struct GETID {
  int operator()(const T& v) {
    return v.getID();
  }
};
// This is a pointer to T specialization!
template <typename T> struct GETID<T*> {
  unsigned operator()(const T* v) {
    return (unsigned)(((uintptr_t) v) >> 7);
  }
};

#ifdef GALOIS_TBB
template<class Compare = std::less<int>, typename T = int>
class PTbb : private boost::noncopyable {
  typedef tbb::concurrent_priority_queue<T,Compare> TBBTy;
  
  struct PTD {
    TBBTy wl;
    GaloisRuntime::LL::PaddedLock<true> L;
    bool V;
    std::vector<T> inq;
  };

  GaloisRuntime::PerCPU<PTD> tld;
  int nactive;

public:
  PTbb() {
    nactive = GaloisRuntime::getSystemThreadPool().getActiveThreads();
  }
  template<bool newconcurrent>
  struct rethread {
    typedef PTbb<Compare, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef PTbb<Compare, Tnew> WL;
  };
  
  typedef T value_type;
  
  bool push(value_type val) {
    unsigned index = GETID<value_type>()(val) % nactive;
    PTD& N = tld.get(index);
    if (index == tld.myEffectiveID())
      N.wl.push(val);
    else {
      N.L.lock();
      N.inq.push_back(val);
      N.V = true;
      N.L.unlock();
    }
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
    return true;
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    while (b != e)
      push(*b++);
  }

  boost::optional<value_type> pop() {
    value_type retval;
    PTD& N = tld.get();
    if (N.V) {
      N.L.lock();
      for (typename std::vector<T>::iterator ii = N.inq.begin(), ee = N.inq.end();
	   ii != ee; ++ii)
	N.wl.push(*ii);
      N.inq.clear();
      N.V = false;
      N.L.unlock();
    }
    boost::optional<value_type> rv;
    if (N.wl.try_pop(retval))
      rv = retval;
    return rv;
  }
};

template<class Compare = std::less<int>, typename T = int>
class TbbPriQueue : private boost::noncopyable {
  tbb::concurrent_priority_queue<T,Compare> wl;

public:
  template<bool newconcurrent>
  struct rethread {
    typedef TbbPriQueue<Compare, T> WL;
  };
  template<typename Tnew>
  struct retype {
    typedef TbbPriQueue<Compare, Tnew> WL;
  };

  typedef T value_type;

  bool push(value_type val) {
    wl.push(val);
    return true;
  }

  template<typename Iter>
  bool push(Iter b, Iter e) {
    while (b != e)
      push(*b++);
    return true;
  }

  template<typename Iter>
  void push_initial(Iter b, Iter e) {
    push(b,e);
  }

  boost::optional<value_type> pop() {
    value_type retval;
    if (wl.try_pop(retval)) {
      return boost::optional<value_type>(retval);
    } else {
      return boost::optional<value_type>();
    }
  }
};
WLCOMPILECHECK(TbbPriQueue);
#endif //TBB

 template<
   typename DefaultWorklist,
   typename dChunk,
   typename Chunk,
   typename Indexer,
   typename Less,
   typename Greater
   >
struct StartWorklistExperiment {
  template<typename Iterator,typename Functor>
  void operator()(std::ostream& out, Iterator ii, Iterator ei, Functor fn) {
    using namespace GaloisRuntime::WorkList;

    typedef OrderedByIntegerMetric<Indexer, dChunk> OBIM;
#ifdef GALOIS_TBB
    typedef TbbPriQueue<Greater> TBB;
    typedef LocalStealing<TBB> LTBB;
    typedef PTbb<Greater> PTBB;
#endif
    typedef SkipListQueue<Less> SLQ;
    typedef SimpleOrderedByIntegerMetric<Indexer> SOBIM;
    typedef LocalStealing<SOBIM> LSOBIM;
    typedef OrderedByIntegerMetric<Indexer, Chunk> NAOBIM;
    typedef BarrierOBIM<Indexer, dChunk> BOBIM;
    typedef CTOrderedByIntegerMetric<Indexer, dChunk> CTOBIM;
    typedef CTOrderedByIntegerMetric<Indexer, Chunk> NACTOBIM;
    typedef LevelStealing<Random<> > RANDOM;

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
    WLFOO(ii, ei, fn, obim,     OBIM)     else
    WLFOO(ii, ei, fn, sobim,    SOBIM)    else
    WLFOO(ii, ei, fn, lsobim,   LSOBIM)   else
    WLFOO(ii, ei, fn, naobim,   NAOBIM)   else
    WLFOO(ii, ei, fn, ctobim,   CTOBIM)   else
    WLFOO(ii, ei, fn, nactobim, NACTOBIM) else
    WLFOO(ii, ei, fn, slq,      SLQ)      else
    WLFOO(ii, ei, fn, bobim,    BOBIM)    else
    WLFOO(ii, ei, fn, bag,      dChunk)   else
    WLFOO(ii, ei, fn, random,   RANDOM)   else
#ifdef GALOIS_TBB
    WLFOO(ii, ei, fn, tbb,      TBB)      else
    WLFOO(ii, ei, fn, ltbb,     LTBB)     else
    WLFOO(ii, ei, fn, ptbb,     PTBB)     else
#endif
    {
      out << "Unrecognized worklist " << name << "\n";
    }
#undef WLFOO
  }
};

} // end namespace
