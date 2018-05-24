/** Wavefront BFS -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Wavefront BFS.
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef WAVEFRONT_BFS_H_
#define WAVEFRONT_BFS_H_

#include "galois/Reduction.h"
#include "galois/AltBag.h"
#include "galois/DoAllWrap.h"
#include "galois/PerThreadContainer.h"

#include "galois/worklists/ExternalReference.h"

#include "bfs.h"

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <set>

class AbstractWavefrontBFS: public BFS {

protected:
  typedef BFS Super_ty;

  //! @return number of iterations
  template <typename WL, typename WFInnerLoop> 
  static size_t runWavefrontBFS (
      Graph& graph, 
      GNode& startNode, 
      const WFInnerLoop& innerLoop) {

    WL* currWL = new WL ();
    WL* nextWL = new WL ();

    graph.getData (startNode, galois::MethodFlag::UNPROTECTED) = 0;
    currWL->push_back (startNode);
    size_t numIter = 1; //  counting the start node

    // while (!currWL->empty ()) {
// 
      // innerLoop (graph, currWL, nextWL);
// 
      // delete currWL;
      // currWL = nextWL;
      // nextWL = new WL ();
    // }
    //

    // since Galois WorkList's do not implement empty () function
    // we rewrite the above loop, counting the number of nodes added to nextWL
    for (bool notDone = true; notDone; ) {

      unsigned numAdds = innerLoop (graph, *currWL, *nextWL);

      notDone = (numAdds > 0);

      delete currWL;
      currWL = nextWL;
      nextWL = new WL ();

      numIter += size_t (numAdds);
    }

    delete currWL;
    delete nextWL;
    currWL = NULL;
    nextWL = NULL;

    return numIter;
  }

};

class BFSserialWavefront: public AbstractWavefrontBFS {

protected:
  typedef galois::SerialBag<GNode> WL_ty;
  typedef AbstractWavefrontBFS::Super_ty Super_ty;


private:
  struct SerialInnerLoop {
    GALOIS_ATTRIBUTE_PROF_NOINLINE unsigned operator () (Graph& graph, WL_ty& currWL, WL_ty& nextWL) const {

      unsigned numAdds = 0;

      for (WL_ty::iterator src = currWL.begin (), esrc = currWL.end ();
          src != esrc; ++src) {

        numAdds += Super_ty::bfsOperator<false> (graph, *src, nextWL);
      }

      return numAdds;
    }
  };



public:
  virtual const std::string getVersion () const { return "Serial Wavefront"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {
    return AbstractWavefrontBFS::runWavefrontBFS<WL_ty> (graph, startNode, SerialInnerLoop ());
  }

};


template <typename WL> 
struct DoAllFunctor {
  typedef BFS BaseBFS;

  typename BaseBFS::Graph& graph;
  WL& nextWL;
  ParCounter& numAdds;

  DoAllFunctor (
      typename BaseBFS::Graph& _graph,
      WL& _nextWL,
      ParCounter& _numAdds)
    :
      graph (_graph),
      nextWL (_nextWL),
      numAdds (_numAdds)
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (typename BaseBFS::GNode& src) {
    numAdds += BFS::bfsOperator<false, WL> (graph, src, nextWL);
  }

};

template <bool doLock> 
struct LoopFlags {
  typedef int tt_does_not_need_stats; // disable stats in galois::runtime
  typedef int tt_does_not_need_push;
};

template <>
struct LoopFlags<false> { // more when no locking
  typedef int tt_does_not_need_stats; // disable stats in galois::runtime
  typedef char tt_does_not_need_push;
  typedef double tt_does_not_need_aborts;
};


template <bool doLock, typename WL>
struct ForEachFunctor: public DoAllFunctor<WL>, public LoopFlags<doLock> {

  typedef DoAllFunctor<WL> Super_ty;
  typedef BFS BaseBFS;

  ForEachFunctor (
      typename BaseBFS::Graph& graph,
      WL& nextWL,
      ParCounter& numAdds)
    :
      Super_ty (graph, nextWL, numAdds) 
  {} 

  template <typename ContextTy>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (typename BaseBFS::GNode src, ContextTy&) {
    Super_ty::operator () (src);
  }
};

template <typename GWL>
struct GaloisWLwrapper: public GWL {

  GaloisWLwrapper (): GWL () {}

  inline void push_back (const typename GWL::value_type& x) {
    GWL::push (x);
  }
};

class BFSwavefrontNolock;

class BFSwavefrontLock: public AbstractWavefrontBFS {
protected:
  typedef GaloisWLwrapper< galois::worklists::dChunkedFIFO<DEFAULT_CHUNK_SIZE, GNode> > GaloisWL;

  typedef AbstractWavefrontBFS::Super_ty BaseBFS;

  friend class BFSwavefrontNolock;

private:
  template <bool doLock>
  struct ParallelInnerLoop {
    GALOIS_ATTRIBUTE_PROF_NOINLINE unsigned operator () (Graph& graph, GaloisWL& currWL, GaloisWL& nextWL) const {
      typedef typename GaloisWL::value_type value_type;
      typedef galois::worklists::ExternalReference<GaloisWL> WL;

      value_type *it = nullptr;

      ParCounter numAdds;

      ForEachFunctor<doLock, GaloisWL> l (graph, nextWL, numAdds);
      galois::for_each(it, it, l, galois::wl<WL>(std::ref(currWL)));

      return numAdds.reduce ();
    }

  };

public:
  virtual const std::string getVersion () const { return "Galois Wavefront with Locking"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {

    
    return AbstractWavefrontBFS::runWavefrontBFS<GaloisWL> (
        graph, startNode, 
        ParallelInnerLoop<true> ()); // true means acquire locks
  }

};

class BFSwavefrontNolock: public AbstractWavefrontBFS {
  virtual const std::string getVersion () const { return "Galois Wavefront NO Locking"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {
    
    return AbstractWavefrontBFS::runWavefrontBFS<BFSwavefrontLock::GaloisWL> (
        graph, startNode, 
        BFSwavefrontLock::ParallelInnerLoop<false> ()); // false for no locking
  }
};


class BFSwavefrontCoupled: public AbstractWavefrontBFS {

// #define BFS_WF_USE_BAG 1

#ifdef BFS_WF_USE_BAG
  typedef galois::InsertBag<GNode> WL_ty;
#else
  typedef galois::PerThreadBag<GNode> WL_ty;
  // typedef galois::PerThreadBag<GNode, 64> WL_ty;
#endif // BFS_WF_USE_BAG

  struct ParallelInnerLoop {
    Graph& graph;
    WL_ty& nextWL;
    ParCounter& numAdds;

    ParallelInnerLoop (
        Graph& graph,
        WL_ty& nextWL,
        ParCounter& numAdds)
      :
        graph (graph),
        nextWL (nextWL),
        numAdds (numAdds) 
    {} 

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (GNode src) const {
#ifdef BFS_WF_USE_BAG
      numAdds += Super_ty::bfsOperator<false> (graph, src, nextWL);
#else
      numAdds += Super_ty::bfsOperator<false> (graph, src, nextWL.get ());
#endif
    }
  };

public:

  virtual const std::string getVersion () const { return "Galois Wavefront Coupled DoAll"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {


    WL_ty* currWL = new WL_ty ();
    WL_ty* nextWL = new WL_ty ();

    graph.getData (startNode, galois::MethodFlag::UNPROTECTED) = 0;
#ifdef BFS_WF_USE_BAG
    currWL->push_back (startNode);
#else
    currWL->get ().push_back (startNode);
#endif

    size_t numIter = 1;

    unsigned level = 0;

    ParCounter numAdds;
    galois::substrate::getThreadPool ().burnPower (galois::getActiveThreads ());

#ifdef BFS_WF_USE_BAG
    while (!currWL->empty ()) {
#else
    while (!currWL->empty_all ()) {
#endif

      galois::do_all_choice (galois::runtime::makeLocalRange(*currWL), 
          ParallelInnerLoop (graph, *nextWL, numAdds), 
          std::make_tuple (
            galois::loopname("wavefront_inner_loop"),
            galois::chunk_size<DEFAULT_CHUNK_SIZE> ()));

      std::swap (currWL, nextWL);

#ifdef BFS_WF_USE_BAG
      nextWL->clear ();
#else
      nextWL->clear_all_parallel ();
#endif
      ++level;
    }
    galois::substrate::getThreadPool ().beKind ();

    numIter += numAdds.reduce ();

    delete currWL; currWL = NULL;
    delete nextWL; nextWL = NULL;

    return numIter;
  }

};




#endif // WAVEFRONT_BFS_H_

