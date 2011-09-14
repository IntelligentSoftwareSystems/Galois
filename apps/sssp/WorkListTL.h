//#include <boost/thread/shared_mutex.hpp>
#include <tbb/concurrent_hash_map.h>

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
