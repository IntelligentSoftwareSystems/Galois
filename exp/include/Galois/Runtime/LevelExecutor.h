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
#include "Galois/Galois.h"
#include "Galois/GaloisUnsafe.h"
#include "Galois/PriorityQueue.h"
#include "Galois/Runtime/UserContextAccess.h"
#include "Galois/WorkList/WorkList.h"
#include "Galois/Runtime/ll/ThreadRWlock.h"
#include "Galois/Runtime/mm/Mem.h"


namespace Galois {
namespace Runtime {

template <typename WL>
class LevelImpl: public WL {
  using Counter = GAccumulator<size_t>;

  mutable Counter sz;

public:

  void push (const typename WL::value_type& x) {
    WL::push (x);
    sz += 1;
  }


  size_t size () const {
    return sz.reduce ();
  }
};

template <typename Key, typename KeyCmp, typename LI>
class LevelWL {

public:
  using value_type = typename LI::value_type;
  using MapAlloc = MM::FSBGaloisAllocator<std::pair<const Key, LI*> >;
  using LevelMap = std::map<Key, LI*, KeyCmp, MapAlloc>;


private:
  
  LL::ThreadRWlock rwmutex;
  LevelMap levelMap;


public:

  LevelWL (const KeyCmp& kcmp): 
    rwmutex (), 
    levelMap (kcmp) 
  {}

  std::pair<const Key, LI*> earliest (void) const {
    assert (!levelMap.empty ());
    return *levelMap.begin ();
  }

  bool empty (void) const {
    return levelMap.empty ();
  }

  size_t size (void) const {
    return levelMap.size ();
  }

  void push (const Key& k, const value_type& x) {

    rwmutex.readLock ();
    auto currLevel = levelMap.find (k);

    if (currLevel == levelMap.end ()) {
      rwmutex.readUnlock (); // give up read lock to acquire write lock

      rwmutex.writeLock (); 
      // check again after locking
      if (levelMap.find (k) == levelMap.end ()) {
        LI* wl = new LI ();
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

  }

  // must be called outside parallel phase
  void removeLevel (const Key& k) {
    auto level = levelMap.find (k);
    assert (level != levelMap.end ());

    levelMap.erase (level);

    // delete wl
    delete level->second;
  }
};



template <typename T, typename Key, typename KeyFn, typename KeyCmp, typename NhoodFunc, typename OpFunc>   
class LevelExecutor {


  // static const unsigned CHUNK_SIZE = 64;
  static const unsigned CHUNK_SIZE = OpFunc::CHUNK_SIZE;

  using WL = Galois::WorkList::dChunkedFIFO<CHUNK_SIZE, T>;
  using LevelImpl_ty = LevelImpl<WL>;
  using LevelWL_ty = LevelWL<Key, KeyCmp, LevelImpl_ty>;

  using UserCtx = UserContextAccess<T>;
  using PerThreadUserCtx = PerThreadStorage<UserCtx>;

  struct BodyWrapper {

    KeyFn& keyFn;
    NhoodFunc& nhVisit;
    OpFunc& opFunc;
    LevelWL_ty& level_wl;
    PerThreadUserCtx& userHandles;

    BodyWrapper (
        KeyFn& keyFn,
        NhoodFunc& nhVisit,
        OpFunc& opFunc,
        LevelWL_ty& level_wl,
        PerThreadUserCtx& userHandles)
      :
        keyFn (keyFn),
        nhVisit (nhVisit),
        opFunc (opFunc),
        level_wl (level_wl),
        userHandles (userHandles)
    {}
    
    template <typename C>
    void operator () (T& x, C& __c) {

      UserCtx& uhand = *userHandles.getLocal ();
      uhand.reset ();
      nhVisit (x, uhand);
      opFunc (x, uhand);

      if (true) { // TODO: change to check for noadd trait
        for (auto i = uhand.getPushBuffer ().begin ()
            , endi = uhand.getPushBuffer ().end (); i != endi; ++i) {

          // using Galois WL, by ref, automatically handles
          // adds to the current level
          level_wl.push (keyFn (*i), *i);
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
  LevelWL_ty level_wl;


public:

  LevelExecutor (
      const KeyFn& keyFn,
      const KeyCmp& kcmp,
      const NhoodFunc& nhVisit,
      const OpFunc& opFunc)
    :
      keyFn (keyFn),
      kcmp (kcmp),
      nhVisit (nhVisit),
      opFunc (opFunc),
      level_wl (kcmp)
  {}

  template <typename I>
  void fill_initial (I beg, I end) {
    Galois::do_all (beg, end,
        [this] (const T& x) {
          level_wl.push (keyFn (x), x);
        });
  }

  void execute () {

    PerThreadUserCtx userHandles;

    size_t steps = 0;
    size_t totalWork = 0;


    while (!level_wl.empty ()) {

      std::pair<const Key, LevelImpl_ty*> currLevel = level_wl.earliest ();
      ++steps;
      totalWork += currLevel.second->size ();

      // std::cout << "Executing " << steps << ", level: " << currLevel.first 
        // << ", with size: " << currLevel.second->size () << ", map size: " << level_wl.size () << std::endl; 

      Galois::for_each_wl (*currLevel.second, 
          BodyWrapper (keyFn, nhVisit, opFunc, level_wl, userHandles), "parallel_level_loop");

      level_wl.removeLevel (currLevel.first);
    }

    std::cout << "Level-by-Level, critical path length: " << steps << ", avg. parallelism: " << ((double) totalWork)/steps << std::endl;
  }
};

template <typename Iter, typename KeyFn, typename KeyCmp, typename NhoodFunc, typename OpFunc>
void for_each_ordered_level (Iter beg, Iter end, const KeyFn& keyFn, const KeyCmp& kcmp, const NhoodFunc& nhVisit, const OpFunc& opFunc, const char* loopname=0) {

  using T = typename std::iterator_traits<Iter>::value_type;
  using K = decltype (keyFn (*beg)); // XXX: check
  using Exec = LevelExecutor<T, K, KeyFn, KeyCmp, NhoodFunc, OpFunc>;

  Exec e (keyFn, kcmp, nhVisit, opFunc);

  e.fill_initial (beg, end);

  e.execute ();

}

} // end namespace Runtime
} // end namespace Galois

#endif // GALOIS_RUNTIME_LEVEL_EXECUTOR_H
