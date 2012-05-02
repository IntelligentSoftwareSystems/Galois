#ifndef GALOIS_RUNTIME_DETERMINISTIC_H
#define GALOIS_RUNTIME_DETERMINISTIC_H

#ifdef GALOIS_DET

#include "Galois/Runtime/ParallelWorkInline.h"

#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

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

template<class T,class FunctionTy>
class DeterministicExecutor {
  typedef T value_type;
  typedef std::pair<T,unsigned long> WorkItem;
  typedef std::pair<WorkItem,SimpleRuntimeContext*> PendingItem;
  typedef std::pair<T,std::pair<unsigned long,unsigned> > NewItem;
  typedef HIDDEN::dChunkedLIFO<WorkItem,256> WL;
  typedef HIDDEN::dChunkedLIFO<PendingItem,256> PendingWork;
  typedef HIDDEN::dChunkedLIFO<NewItem,256> NewWork;

  // Truly thread-local
  struct ThreadLocalData: private boost::noncopyable {
    GaloisRuntime::UserContextAccess<value_type> facing;
    LoopStatistics<ForEachTraits<FunctionTy>::NeedsStats> stat;
    HIDDEN::WID wid;
    ContextPool pool;
    size_t newSize;
    WL* wlcur;
    WL* wlnext;
    ThreadLocalData() { reset(); }
    void reset() {
      newSize = 0;
    }
  };

  // Mostly local but shared sometimes
  struct MergeInfo {
    typedef std::vector<NewItem> Data;
    Data data;
    size_t begin;
    size_t newItems;
    size_t committed;
    size_t nextMax;
    size_t skipped;
    MergeInfo(): nextMax(-1) { reset(); }
    void reset() {
      newItems = 0;
      committed = 0;
      skipped = 0;
      data.clear();
    }
  };

  typedef typename MergeInfo::Data MergeInfoData;
  typedef typename MergeInfoData::iterator MergeInfoDataIterator;

  //! Flatten multiple MergeInfos into single iteration domain
  struct MergeIterator:
    public boost::iterator_facade<MergeIterator,
                                  NewItem,
                                  boost::forward_traversal_tag> {
    PerCPU<MergeInfo>* base;
    int oo, eo;
    MergeInfoDataIterator ii, ei;

    void increment() {
      ++ii;
      while (ii == ei) {
        if (++oo == eo)
          break;

        ii = base->get(oo).data.begin();
        ei = base->get(oo).data.end();
      }
    }

    bool equal(const MergeIterator& x) const {
      // end pointers?
      if (oo == eo)
        return x.oo == x.eo;
      else if (x.oo == x.eo)
        return oo == eo;
      else
        return oo == x.oo && ii == x.ii;
    }

    NewItem& dereference() const {
      return *ii;
    }

    void initRange(PerCPU<MergeInfo>* _base, int b, int e, MergeIterator& end);
  };

  PerCPU<MergeInfo> mergeInfo;
  typename MergeInfo::Data mergeBuf;
  GBarrier barrier[4];
  WL wl[2];
  NewWork new_;
  PendingWork pending;
  FunctionTy& function;
  const char* loopname;
  LL::CacheLineStorage<volatile long> done;
  int numActive;

  bool merge(int begin, int end);
  bool checkEmpty(ThreadLocalData& tld);
  bool renumber(ThreadLocalData& tld);
  bool renumberMaster();
  void pendingLoop(ThreadLocalData& tld);
  void commitLoop(ThreadLocalData& tld);

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

protected:
  void go();

public:
  DeterministicExecutor(FunctionTy& f, const char* ln): function(f), loopname(ln) { 
    numActive = (int) GaloisRuntime::getSystemThreadPool().getActiveThreads();
    for (int i = 0; i < 4; ++i)
      barrier[i].reinit(numActive);
    if (ForEachTraits<FunctionTy>::NeedsBreak) abort();
  }

  ~DeterministicExecutor() {
    if (ForEachTraits<FunctionTy>::NeedsStats)
      GaloisRuntime::statDone();
  }

  template<typename IterTy>
  bool AddInitialWork(IterTy b, IterTy e) {
    if (LL::getTID() != 0)
      return true;

    HIDDEN::WID wid;
    WorkItem a[1];
    unsigned long id = 0;
    while (b != e) {
      a[0] = std::make_pair(*b, ++id);
      wl[0].push_initial(wid, &a[0], &a[1]);
      ++b;
    }
    return true;
  }
};

template<class T, class FunctionTy>
class ForEachWork<WorkList::Deterministic<>,T,FunctionTy>: public DeterministicExecutor<T,FunctionTy>
{
  typedef DeterministicExecutor<T,FunctionTy> Super;
public:
  ForEachWork(FunctionTy& f, const char* loopname): Super(f, loopname) { }

  template<typename IterTy>
  bool AddInitialWork(IterTy b, IterTy e) {
    return Super::AddInitialWork(b, e);
  }

  void operator()() {
    Super::go();
  }
};

template<class T,class FunctionTy>
void DeterministicExecutor<T,FunctionTy>::MergeIterator::initRange(PerCPU<MergeInfo>* _base, int b, int e, MergeIterator& end) {
  base = _base;
  oo = b;
  eo = e;

  end.base = _base;
  end.oo = e;
  end.eo = e;

  if (oo != eo) {
    ii = base->get(oo).data.begin();
    ei = base->get(oo).data.end();
    while (ii == ei) {
      if (++oo == eo)
        break;
      ii = base->get(oo).data.begin();
      ei = base->get(oo).data.end();
    }
  }
}

template<class T,class FunctionTy>
void DeterministicExecutor<T,FunctionTy>::go() {
  ThreadLocalData tld;
  tld.wlcur = &wl[0];
  tld.wlnext = &wl[1];

  while (true) {
    setPending(PENDING);
    pendingLoop(tld);

    barrier[0].wait();

    setPending(COMMITTING);
    commitLoop(tld);

    barrier[1].wait();

    tld.pool.commit();

    if (ForEachTraits<FunctionTy>::NeedsPush) {
      if (renumber(tld))
        break;
    } else {
      if (checkEmpty(tld))
        break;
      std::swap(tld.wlcur, tld.wlnext);
    }
  }
  setPending(NON_DET);
  if (ForEachTraits<FunctionTy>::NeedsStats)
    tld.stat.report_stat(LL::getTID(), loopname);
}

template<class T,class FunctionTy>
bool DeterministicExecutor<T,FunctionTy>::merge(int begin, int end)
{
  if (begin == end)
    return false;
  else if (begin + 1 == end)
    return !mergeInfo.get(begin).data.empty();
  
  bool retval = false;
  unsigned mid = (end - begin) / 2 + begin;
  retval |= merge(begin, mid);
  retval |= merge(mid, end);

  LessThan lessThan;
  MergeIterator aa, ea, bb, eb, cur, ecur;

  aa.initRange(&mergeInfo, begin, mid, ea);
  bb.initRange(&mergeInfo, mid, end, eb);
  cur.initRange(&mergeInfo, begin, end, ecur);

  while (aa != ea && bb != eb) {
    if (lessThan(*aa, *bb))
      mergeBuf.push_back(*aa++);
    else
      mergeBuf.push_back(*bb++);
  }

  for (; aa != ea; ++aa)
    mergeBuf.push_back(*aa);

  for (; bb != eb; ++bb)
    mergeBuf.push_back(*bb);

  for (MergeInfoDataIterator ii = mergeBuf.begin(), ei = mergeBuf.end(); ii != ei; ++ii) 
    *cur++ = *ii; 
  mergeBuf.clear();

  assert(cur == ecur);

  return retval;
}

template<class T,class FunctionTy>
bool DeterministicExecutor<T,FunctionTy>::checkEmpty(ThreadLocalData& tld)
{
  if (tld.wid.tid == 0) {
    bool empty = true;
    for (int i = 0; i < numActive; ++i) {
      HIDDEN::WID wid(i);
      if (!tld.wlnext->empty(wid)) {
        empty = false;
        break;
      }
    }

    done.data = empty;
  }
  
  barrier[2].wait();

  return done.data;
}

template<class T,class FunctionTy>
bool DeterministicExecutor<T,FunctionTy>::renumberMaster()
{
  // Accumulate all threads' info
  size_t begin = 0;
  size_t newItems = 0;
  size_t committed = 0;
  size_t skipped = 0;
  for (int i = 0; i < numActive; ++i) {
    MergeInfo& minfo = mergeInfo.get(i);
    minfo.begin = begin;

    size_t newSize = minfo.data.size();
    newItems += minfo.newItems;
    committed += minfo.committed;
    skipped += minfo.skipped;
    begin += newSize;
  }

  // Compute new window size
  const float target = 0.90;
  size_t total = begin - newItems - skipped + committed;
  float commitRatio = total > 0 ? committed / (float) total : 0.0;

  size_t prevMax = mergeInfo.get(0).nextMax;
  size_t nextMax;
  if (commitRatio < target) {
    nextMax = commitRatio / target * prevMax; //begin;
  } else {
    nextMax = 2*prevMax;
  }
  if (nextMax == 0) nextMax = 1;
  if (nextMax > begin) nextMax = begin;

  printf("R %ld %.3f %zu t:%zu c:%zu %zu %zu\n", 
      begin, commitRatio, nextMax, total, committed, newItems, skipped);

  // Tell everyone new window size
  for (int i = 0; i < numActive; ++i) {
    MergeInfo& minfo = mergeInfo.get(i);
    //minfo.nextMax = nextMax;
  }

  // Prepare for merge
  mergeBuf.reserve(begin);

  return !merge(0, numActive);
}

template<class T,class FunctionTy>
bool DeterministicExecutor<T,FunctionTy>::renumber(ThreadLocalData& tld)
{
#if 0
  MergeInfo& minfo = mergeInfo.get();

  minfo.data.reserve(tld.newSize * 2);
  while (!new_.empty(tld.wid)) {
    NewItem& p = new_.back(tld.wid);
    minfo.data.push_back(p);
    new_.pop_back(tld.wid);
  }

  std::sort(minfo.data.begin(), minfo.data.end(), LessThan());
  
  barrier[2].wait();

  if (tld.wid.tid == 0) {
    done.data = renumberMaster();
  }

  barrier[3].wait();

  // TODO: shuffle blocks among threads
  size_t id = minfo.begin;
  size_t nextMax = minfo.nextMax;
  size_t newSize = 0;
  size_t skipped = 0;
  for (MergeInfoDataIterator ii = minfo.data.begin(), ei = minfo.data.end(); ii != ei; ++ii) {
    if (++id <= nextMax) {
      tld.wlcur->push_back(tld.wid, std::make_pair(ii->first, id));
    } else {
      new_.push_back(tld.wid, std::make_pair(ii->first, std::make_pair(id, 0)));
      ++newSize;
      ++skipped;
    }
  }

  minfo.reset();
  minfo.skipped += skipped;
  tld.reset();
  tld.newSize += newSize;

  return done.data;
#else
  if (LL::getTID() == 0) {
    std::vector<NewItem> buf;
    for (int i = 0; i < numActive; ++i) {
      HIDDEN::WID wid(i);
      while (!new_.empty(wid)) {
        NewItem& p = new_.back(wid);
        buf.push_back(p);
        new_.pop_back(wid);
      }
    }

    std::sort(buf.begin(), buf.end(), LessThan());
    typename std::vector<NewItem>::iterator uni = std::unique(buf.begin(), buf.end(), EqualTo());
    assert(uni == buf.end());

    unsigned long id = 0;
    done.data = buf.empty();

    printf("R %ld\n", buf.size());

    HIDDEN::WID wid;
    for (typename std::vector<NewItem>::iterator ii = buf.begin(), ei = buf.end(); ii != ei; ++ii) {
      tld.wlcur->push_back(wid, std::make_pair(ii->first, id++));
    }
  }

  barrier[2].wait();

  return done.data;
#endif
}

template<class T,class FunctionTy>
void DeterministicExecutor<T,FunctionTy>::pendingLoop(ThreadLocalData& tld)
{
  SimpleRuntimeContext* cnx = tld.pool.next();
  setThreadContext(cnx);

  while (!tld.wlcur->empty(tld.wid)) {
    WorkItem& p = tld.wlcur->back(tld.wid);

    bool commit = true;
    cnx->setId(p.second);
    cnx->start_iteration();
    tld.stat.inc_iterations();
    try {
      function(p.first, tld.facing.data());
    } catch (ConflictFlag i) {
      switch (i) {
        case CONFLICT: commit = false; break;
        case REACHED_FAILSAFE: break;
        default: assert(0 && "Unknown conflict flag"); abort(); break;
      }
    }

    if (ForEachTraits<FunctionTy>::NeedsPIA)
      tld.facing.resetAlloc();

    if (commit) {
      pending.push_back(tld.wid, std::make_pair(p, cnx));
    } else if (ForEachTraits<FunctionTy>::NeedsPush) {
      new_.push_back(tld.wid, std::make_pair(p.first, std::make_pair(p.second, 0)));
      ++tld.newSize;
      tld.stat.inc_conflicts();
    } else {
      tld.wlnext->push_back(tld.wid, p);
      tld.stat.inc_conflicts();
    }

    cnx = tld.pool.next();
    setThreadContext(cnx);
    tld.wlcur->pop_back(tld.wid);
  }
}

template<class T,class FunctionTy>
void DeterministicExecutor<T,FunctionTy>::commitLoop(ThreadLocalData& tld) 
{
  MergeInfo& minfo = mergeInfo.get();

  while (!pending.empty(tld.wid)) {
    PendingItem& p = pending.back(tld.wid);
    bool commit = true;
    try {
      setThreadContext(p.second);
      function(p.first.first, tld.facing.data());
    } catch (ConflictFlag i) {
      switch (i) {
        case CONFLICT: commit = false; break;
        default: assert(0 && "Unknown exception"); abort(); break;
      }
    }
    
    if (commit) {
      ++minfo.committed;
      unsigned long parent = p.first.second;
      typedef typename GaloisRuntime::UserContextAccess<value_type>::pushBufferTy::iterator iterator;
      if (ForEachTraits<FunctionTy>::NeedsPush) {
        unsigned count = 0;
        for (iterator ii = tld.facing.getPushBuffer().begin(), 
            ei = tld.facing.getPushBuffer().end(); ii != ei; ++ii) {
          new_.push_back(tld.wid, std::make_pair(*ii, std::make_pair(parent, ++count)));
          ++tld.newSize;
          ++minfo.newItems;
          if (count == 0) {
            assert(0 && "Counter overflow");
            abort();
          }
        }
      }
      assert(ForEachTraits<FunctionTy>::NeedsPush
          || tld.facing.getPushBuffer().begin() == tld.facing.getPushBuffer().end());
    } else if (ForEachTraits<FunctionTy>::NeedsPush) {
      new_.push_back(tld.wid, std::make_pair(p.first.first, std::make_pair(p.first.second, 0)));
      ++tld.newSize;
      tld.stat.inc_conflicts();
    } else {
      tld.wlnext->push_back(tld.wid, p.first);
      tld.stat.inc_conflicts();
    }

    if (ForEachTraits<FunctionTy>::NeedsPIA)
      tld.facing.resetAlloc();

    tld.facing.resetPushBuffer();

    pending.pop_back(tld.wid);
  }

  setThreadContext(0);
}

}

#endif

#endif
