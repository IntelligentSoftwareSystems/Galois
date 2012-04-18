#ifndef GALOIS_RUNTIME_DETERMINISTIC_H
#define GALOIS_RUNTIME_DETERMINISTIC_H

#ifdef GALOIS_DET

#include "Galois/Runtime/ParallelWorkInline.h"

namespace GaloisRuntime {

class ContextPool: private boost::noncopyable {
  template<typename T> struct LinkedNode { void *x,*y; T z; };
  typedef Galois::GFixedAllocator<LinkedNode<SimpleRuntimeContext> > Allocator;
  typedef std::list<SimpleRuntimeContext,Allocator> List;
  typedef List::iterator iterator;

  List list;
  iterator end;

public:
  ContextPool(): end(list.end()) { }

  SimpleRuntimeContext* next() {
    SimpleRuntimeContext* r;
    if (end != list.end()) {
      r = &*end;
      ++end;
    } else {
      list.push_back(SimpleRuntimeContext());
      r = &list.back();
      end = list.end();
    }
    return r;
  }

  void commit() {
    for (iterator ii = list.begin(); ii != end; ++ii)
      ii->commit_iteration();
    end = list.begin();
  }
};

template<class T, class FunctionTy, bool isSimple>
class ForEachWork<WorkList::Deterministic<>,T,FunctionTy,isSimple>: public ForEachWorkBase<T,  FunctionTy> {
  typedef ForEachWorkBase<T, FunctionTy> Super;
  typedef typename Super::value_type value_type;
  typedef typename Super::ThreadLocalData ThreadLocalData;
  typedef std::pair<T,unsigned long> WorkItem;
  typedef std::pair<WorkItem,SimpleRuntimeContext*> PendingItem;
  typedef std::pair<T,std::pair<unsigned long,unsigned> > NewItem;
  typedef HIDDEN::dChunkedLIFO<WorkItem,256> WL;
  typedef HIDDEN::dChunkedLIFO<PendingItem,256> Pending;
  typedef HIDDEN::dChunkedLIFO<NewItem,256> New;

  PerCPU<ContextPool> pool;
  FastBarrier barrier1;
  FastBarrier barrier2;
  FastBarrier barrier3;
  WL wl;
  Pending pending;
  New new_;
  LL::CacheLineStorage<volatile long> done;
  unsigned numActive;

  bool renumber();
  void pendingLoop(const HIDDEN::WID& wid, ThreadLocalData& tld);
  void commitLoop(const HIDDEN::WID& wid, ThreadLocalData& tld);
  void go();

  struct LessThan {
    bool operator()(const NewItem& a, const NewItem& b) const {
      if (a.second.first < b.second.first)
        return true;
      else if (a.second.first == b.second.first) 
        return a.second.second < b.second.second;
      else
        return false;
    }
  };

  struct EqualTo {
    bool operator()(const NewItem& a, const NewItem& b) const {
      return a.second.first == b.second.first && a.second.second == b.second.second;
    }
  };

public:
  ForEachWork(FunctionTy& f, const char* loopname): Super(f, loopname) { 
    numActive = GaloisRuntime::getSystemThreadPool().getActiveThreads();
    barrier1.reinit(numActive);
    barrier2.reinit(numActive);
    barrier3.reinit(numActive);
    // TODO(ddn): support break
    if (ForeachTraits<FunctionTy>::NeedsBreak) abort();
  }

  template<typename IterTy>
  bool AddInitialWork(IterTy b, IterTy e) {
    HIDDEN::WID wid;
    WorkItem a[1];
    unsigned long id = 0;
    while (b != e) {
      a[0] = std::make_pair(*b, ++id);
      wl.push_initial(wid, &a[0], &a[1]);
      ++b;
    }
    return true;
  }

  bool parallelInitialWork() const {
    return false;
  }

  void operator()() {
    go();
  }
};

template<class T,class FunctionTy,bool isSimple>
void ForEachWork<WorkList::Deterministic<>,T,FunctionTy,isSimple>::go() {
  ThreadLocalData& tld = Super::initWorker();
  unsigned tid = LL::getTID();

  HIDDEN::WID wid;

  while (true) {
    setPending(true);
    pendingLoop(wid, tld);

    barrier1.wait();

    setPending(false);
    commitLoop(wid, tld);

    barrier2.wait();

    pool.get(wid.tid).commit();
    // TODO parallelize renumber
    // TODO: !needsPush specialization
    if (tid == 0) {
      if (!renumber())
        done.data = true;
    }
    barrier3.wait();

    if (done.data)
      break;
  }
  Super::deinitWorker();
}

template<class T,class FunctionTy,bool isSimple>
bool ForEachWork<WorkList::Deterministic<>,T,FunctionTy,isSimple>::renumber()
{
  std::vector<NewItem> buf;
  for (unsigned i = 0; i < numActive; ++i) {
    HIDDEN::WID wid(i);
    while (!new_.empty(wid)) {
      NewItem& p = new_.back(wid);
      buf.push_back(p);
      new_.pop_back(wid);
    }
  }

  std::sort(buf.begin(), buf.end(), LessThan());

  unsigned long id = 0;
  bool retval = !buf.empty();

  //printf("Round %zu\n", buf.size());
  HIDDEN::WID wid;
  for (typename std::vector<NewItem>::iterator ii = buf.begin(), ei = buf.end(); ii != ei; ++ii) {
    wl.push_back(wid, std::make_pair(ii->first, id++));
  }

  return retval;
}

template<class T,class FunctionTy,bool isSimple>
void ForEachWork<WorkList::Deterministic<>,T,FunctionTy,isSimple>::pendingLoop(
    const HIDDEN::WID& wid,
    ThreadLocalData& tld)
{
  SimpleRuntimeContext* cnx = pool.get(wid.tid).next();
  setThreadContext(cnx);

  while (!wl.empty(wid)) {
    WorkItem& p = wl.back(wid);
    bool commit = true;
    try {
      cnx->setId(p.second);
      cnx->start_iteration();
      function(p.first, tld.facing);
    } catch (ConflictFlag i) {
      switch (i) {
        case CONFLICT: commit = false; break;
        case REACHED_FAILSAFE: break;
        default: assert(0 && "Unknown conflict flag"); abort(); break;
      }
    }

    if (tld.facing.__getPushBuffer().begin() != tld.facing.__getPushBuffer().end()) {
      assert(0 && "Pushed before failsafe");
      abort();
    }

    Super::incrementIterations(tld);

    if (ForeachTraits<FunctionTy>::NeedsPIA)
      tld.facing.__resetAlloc();

    if (commit) {
      pending.push_back(wid, std::make_pair(p, cnx));
    } else {
      new_.push_back(wid, std::make_pair(p.first, std::make_pair(p.second, 0)));
      Super::incrementConflicts(tld);
    }

    wl.pop_back(wid);
    cnx = pool.get(wid.tid).next();
    setThreadContext(cnx);
  }
}

template<class T,class FunctionTy,bool isSimple>
void ForEachWork<WorkList::Deterministic<>,T,FunctionTy,isSimple>::commitLoop(
    const HIDDEN::WID& wid,
    ThreadLocalData& tld) 
{
  while (!pending.empty(wid)) {
    PendingItem& p = pending.back(wid);
    bool commit = true;
    try {
      setThreadContext(p.second);
      function(p.first.first, tld.facing);
    } catch (ConflictFlag i) {
      switch (i) {
        case CONFLICT: commit = false; break;
        default: assert(0 && "Unknown exception"); abort(); break;
      }
    }
    
    if (commit) {
      if (ForeachTraits<FunctionTy>::NeedsPush) {
        unsigned long parent = p.first.second;
        unsigned count = 0;
        typedef typename Galois::UserContext<value_type>::pushBufferTy::iterator iterator;
        for (iterator ii = tld.facing.__getPushBuffer().begin(), 
            ei = tld.facing.__getPushBuffer().end(); ii != ei; ++ii) {
          new_.push_back(wid, std::make_pair(*ii, std::make_pair(parent, ++count)));
          if (count == 0) {
            assert(0 && "Counter overflow");
            abort();
          }
        }
      }
    } else {
      new_.push_back(wid, std::make_pair(p.first.first, std::make_pair(p.first.second, 0)));
      Super::incrementConflicts(tld);
    }

    if (ForeachTraits<FunctionTy>::NeedsPIA)
      tld.facing.__resetAlloc();

    tld.facing.__resetPushBuffer();

    pending.pop_back(wid);
  }

  setThreadContext(0);
}

}

#endif

#endif
