/** Level-by-Level executor -*- C++ -*-
 * @file
 * This is the only file to include for basic Galois functionality.
 *
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE. NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#ifndef GALOIS_RUNTIME_LEVEL_EXECUTOR_H
#define GALOIS_RUNTIME_LEVEL_EXECUTOR_H

#include <iostream>
#include <map>
#include <vector>


#include "Galois/Accumulator.h"
#include "Galois/optional.h"
#include "Galois/Galois.h"
#include "Galois/GaloisUnsafe.h"
#include "Galois/PriorityQueue.h"
#include "Galois/Runtime/UserContextAccess.h"
#include "Galois/WorkList/WorkList.h"
#include "Galois/WorkList/WorkListWrapper.h"
#include "Galois/Runtime/ll/ThreadRWlock.h"
#include "Galois/Runtime/mm/Mem.h"


namespace Galois {
namespace Runtime {
namespace impl {

#define USE_DENSE_LEVELS 1
// #undef USE_DENSE_LEVELS
//

#define USE_LEVEL_CACHING 1
// #undef USE_LEVEL_CACHING

template <typename Key, typename KeyCmp, typename WL_ty>
class LevelMap {

public:
  using value_type = typename WL_ty::value_type;
  using MapAlloc = MM::FSBGaloisAllocator<std::pair<const Key, WL_ty*> >;
  using WLalloc = MM::FSBGaloisAllocator<WL_ty>;
  using InternalMap = std::map<Key, WL_ty*, KeyCmp, MapAlloc>;
  using GarbageVec = Galois::gdeque<WL_ty*>;
  using Level = std::pair<Key, WL_ty*>;
  using CachedLevel = Galois::optional<Level>;


private:
  
  LL::ThreadRWlock rwmutex;
  InternalMap levelMap;
  WLalloc wlAlloc;
  GarbageVec removedLevels;
  PerThreadStorage<CachedLevel> cachedLevels;

public:

  LevelMap (const KeyCmp& kcmp): 
    rwmutex (), 
    levelMap (kcmp) 
  {
  }

  bool empty (void) const {
    return levelMap.empty ();
  }

  std::pair<const Key, WL_ty*> pop (void) {
    assert (!levelMap.empty ());
    auto p =  *levelMap.begin ();

    levelMap.erase (levelMap.begin ());
    removedLevels.push_back (p.second);

    return p;
  }

  // only push called in parallel by multiple threads
  void push (const Key& k, const value_type& x) {

#ifdef USE_LEVEL_CACHING
    CachedLevel& cached = *(cachedLevels.getLocal ());

    if (cached && k == cached->first) { // fast path
      assert (cached->second != nullptr);
      cached->second->push (x);

    } else  {
#else
    {
#endif

      // debug
      // std::printf ("could not find cached worklist");

      rwmutex.readLock ();
      auto currLevel = levelMap.find (k);

      if (currLevel == levelMap.end ()) {
        rwmutex.readUnlock (); // give up read lock to acquire write lock

        rwmutex.writeLock (); 
        // check again after locking
        if (levelMap.find (k) == levelMap.end ()) {
          WL_ty* wl = wlAlloc.allocate (1); // new WL_ty ();
          new (wl) WL_ty;
          levelMap.insert (std::make_pair (k, wl));
        }
        rwmutex.writeUnlock ();

        // read again now
        rwmutex.readLock ();
        currLevel = levelMap.find (k);
      }
      rwmutex.readUnlock ();

      assert (currLevel != levelMap.end ());
      currLevel->second->push (x);
#ifdef USE_LEVEL_CACHING
      cached = *currLevel;
#endif
    }
    

  }


  void freeRemovedLevels () {
    while (!removedLevels.empty ()) {
      WL_ty* wl = removedLevels.front ();
      removedLevels.pop_front ();
      wlAlloc.destroy (wl);
      wlAlloc.deallocate (wl, 1);

    }
  }
};


#ifdef USE_DENSE_LEVELS

template <typename WL_ty>
class LevelMap<unsigned, std::less<unsigned>, WL_ty> {

public:
  using value_type = typename WL_ty::value_type;
  using WLalloc = MM::FSBGaloisAllocator<WL_ty>;
  using GarbageVec = Galois::gdeque<WL_ty*>;
  using Level = std::pair<unsigned, WL_ty*>;
  using InternalMap = std::deque<Level>;
  using CachedLevel = Galois::optional<Level>;

private:

  LL::ThreadRWlock rwmutex;
  InternalMap levelMap;
  WLalloc wlAlloc;
  GarbageVec removedLevels;
  PerThreadStorage<CachedLevel> cachedLevels;

  unsigned begLevel = 0;

public:

  LevelMap (const std::less<unsigned>&) {}

  bool empty (void) const {
    return levelMap.empty ();
  }

  std::pair<unsigned, WL_ty*> pop (void) {
    assert (!levelMap.empty ());

    const unsigned BAD_VAL = 1 << 30;
    Level p = std::make_pair (BAD_VAL, nullptr);

    while (!levelMap.empty ()) {
      p = levelMap.front ();
      levelMap.pop_front ();
      ++begLevel;
      assert (p.second != nullptr);
      removedLevels.push_back (p.second);

      if (p.second->size () != 0) {
        break;
      }
    }

    return p;
  }

  void push (const unsigned k, const value_type& x) {

#ifdef USE_LEVEL_CACHING
    if (k < begLevel) {
      GALOIS_DIE ("Can't handle non-monotonic adds");
    }

    CachedLevel& cached = *(cachedLevels.getLocal ());

    if (cached && k == cached->first) {
      assert (cached->second != nullptr);
      cached->second->push (x);

    } else {
#else
    {
#endif

      const unsigned index = k - begLevel;

      rwmutex.readLock ();
      if (index >= levelMap.size ()) {
        rwmutex.readUnlock (); // give up read lock to acquire write lock

        rwmutex.writeLock ();
        // check again
        if (index >= levelMap.size ()) {
          // levelMap.resize ((index + 1), std::make_pair (BAD_VAL, nullptr));

          // resize
          for (unsigned i = levelMap.size (); i <= index; ++i) {

            WL_ty* wl = wlAlloc.allocate (1);
            assert (wl != nullptr);
            new (wl) WL_ty;

            levelMap.push_back (std::make_pair (i + begLevel, wl));
          }
          
        }
        rwmutex.writeUnlock ();

        // read again now
        rwmutex.readLock ();

      }

      assert (levelMap.size () > index);
      auto currLevel = levelMap[index];
      assert (currLevel.first == k);
      rwmutex.readUnlock ();

      currLevel.second->push (x);
#ifdef USE_LEVEL_CACHING
      cached = currLevel;
#endif
      
    }
  }
  
  void freeRemovedLevels () {
    while (!removedLevels.empty ()) {
      WL_ty* wl = removedLevels.front ();
      removedLevels.pop_front ();
      wlAlloc.destroy (wl);
      wlAlloc.deallocate (wl, 1);

    }
  }

};

#endif // USE_DENSE_LEVELS


template <bool CanAbort> 
struct InheritTraits {
  typedef char tt_does_not_need_push;
};

template <> 
struct InheritTraits<false> {
  typedef int tt_does_not_need_aborts;
  typedef char tt_does_not_need_push;
};


} // end namespace impl


template <typename T, typename Key, typename KeyFn, typename KeyCmp, typename NhoodFunc, typename OpFunc>   
class LevelExecutor {

  static const unsigned CHUNK_SIZE = OpFunc::CHUNK_SIZE;

  // hack to get ChunkedMaster base class
  using BaseWL = typename Galois::WorkList::dChunkedFIFO<CHUNK_SIZE>::template retype<T>::type;
  using WL_ty = Galois::WorkList::WLsizeWrapper<BaseWL>;
  using LevelMap_ty = impl::LevelMap<Key, KeyCmp, WL_ty>;

  using UserCtx = UserContextAccess<T>;
  using PerThreadUserCtx = PerThreadStorage<UserCtx>;

  struct BodyWrapper;
  using ForEachExec_ty = Galois::Runtime::ForEachWork<WorkList::ExternPtr<WL_ty>, T, BodyWrapper>;



  struct BodyWrapper: public impl::InheritTraits<ForEachTraits<OpFunc>::NeedsAborts> {


    KeyFn& keyFn;
    NhoodFunc& nhVisit;
    OpFunc& opFunc;
    LevelMap_ty& level_map;
    PerThreadUserCtx& userHandles;

    BodyWrapper (
        KeyFn& keyFn,
        NhoodFunc& nhVisit,
        OpFunc& opFunc,
        LevelMap_ty& level_map,
        PerThreadUserCtx& userHandles)
      :
        keyFn (keyFn),
        nhVisit (nhVisit),
        opFunc (opFunc),
        level_map (level_map),
        userHandles (userHandles)
    {}
    
    template <typename C>
    void operator () (T& x, C&) {

      UserCtx& uhand = *userHandles.getLocal ();

      if (ForEachTraits<OpFunc>::NeedsPush) {
        uhand.reset ();
      }

      nhVisit (x, uhand);
      opFunc (x, uhand);

      if (ForEachTraits<OpFunc>::NeedsPush) { // TODO: change to check for noadd trait
        for (auto i = uhand.getPushBuffer ().begin ()
            , endi = uhand.getPushBuffer ().end (); i != endi; ++i) {

          // using Galois WL, by ref, automatically handles
          // adds to the current level
          level_map.push (keyFn (*i), *i);
        }
      }
      uhand.reset ();
    }
  };

private:

  KeyFn keyFn;
  KeyCmp kcmp;
  NhoodFunc nhVisit;
  OpFunc opFunc;
  LevelMap_ty level_map;
  WL_ty* dummy;
  PerThreadUserCtx userHandles;
  ForEachExec_ty for_each_exec;
  Barrier& barrier;

  volatile bool done = false;

public:

  LevelExecutor (
      const KeyFn& _keyFn,
      const KeyCmp& _kcmp,
      const NhoodFunc& _nhVisit,
      const OpFunc& _opFunc,
      const char* loopname)
    :
      keyFn (_keyFn),
      kcmp (_kcmp),
      nhVisit (_nhVisit),
      opFunc (_opFunc),
      level_map (kcmp),
      dummy (new WL_ty ()),
      for_each_exec (
          Galois::WorkList::ExternPtr<WL_ty> (dummy),
          BodyWrapper ( keyFn, nhVisit, opFunc, level_map, userHandles),
          loopname),
      barrier (getSystemBarrier ())

  {}

  ~LevelExecutor (void) {
    delete dummy; dummy = nullptr;
  }

  // parallel
  template <typename R>
  void fill_initial (const R& range) {

    auto rp = range.local_pair ();
    for (auto i = rp.first, i_end = rp.second; i != i_end; ++i) {

      level_map.push (keyFn (*i), *i);
    }

    // Galois::do_all (beg, end,
        // [this] (const T& x) {
          // level_map.push (keyFn (x), x);
        // });
  }

  static bool isMasterThread (void) {
    return LL::getTID () == 0;
  }

  // parallel
  void go (void) {

    unsigned steps = 0;
    size_t totalWork = 0;

    while (true) {
      
      if (isMasterThread ()) {
        level_map.freeRemovedLevels ();
        if (level_map.empty ()) {
          done = true;

        } else {

          std::pair<const Key, WL_ty*> currLevel = level_map.pop ();
          ++steps;
          totalWork += currLevel.second->size ();

          for_each_exec.reinit (typename Galois::WorkList::ExternPtr<WL_ty> (currLevel.second));
        }
      }

      for_each_exec.initThread ();

      barrier.wait ();

      if (done) {
        break;
      }

      for_each_exec ();

      barrier.wait ();

    }

    if (isMasterThread ()) {
      std::cout << "Level-by-Level, critical path length: " << steps << ", avg. parallelism: " << ((double) totalWork)/steps << std::endl;
    }

  }

  void operator () (void) {
    go ();
  }


};

template <typename R, typename KeyFn, typename KeyCmp, typename NhoodFunc, typename OpFunc>
void for_each_ordered_level (const R& range, const KeyFn& keyFn, const KeyCmp& kcmp, const NhoodFunc& nhVisit, const OpFunc& opFunc, const char* loopname=0) {

  using T = typename R::value_type;
  using K = decltype (keyFn (*(range.begin ()))); // XXX: check
  using Exec_ty = LevelExecutor<T, K, KeyFn, KeyCmp, NhoodFunc, OpFunc>;

  Exec_ty exec (keyFn, kcmp, nhVisit, opFunc, loopname);
  Barrier& barrier = getSystemBarrier ();

  if (inGaloisForEach) {
    GALOIS_DIE ("Nested parallelism not supported yet");
  }

  inGaloisForEach = true;

  getSystemThreadPool ().run (Galois::getActiveThreads (),
    std::bind (&Exec_ty::template fill_initial<R>, std::ref (exec), range),
    std::ref (barrier),
    std::ref (exec));

  inGaloisForEach = false;

}

} // end namespace Runtime
} // end namespace Galois

#endif // GALOIS_RUNTIME_LEVEL_EXECUTOR_H
