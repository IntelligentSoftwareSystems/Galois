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

#include <string>
#include <sstream>
#include <limits>
#include <iostream>
#include <set>

#include "Galois/GaloisUnsafe.h"
#include "Galois/Accumulator.h"

#include "Galois/Runtime/PerThreadWorkList.h"
#include "Galois/Runtime/DoAll.h"

#include "bfs.h"

// vtune
#ifdef GALOIS_USE_VTUNE
#include "ittnotify.h"
#endif

// TODO: write a LCGraph without edge data
// TODO: allow passing worklist ref to for_each


// can't use void for edge data type, therefore, using unsigned
typedef Galois::Graph::LC_CSR_Graph<unsigned, unsigned> Graph;
typedef Graph::GraphNode GNode;

typedef Galois::GSimpleReducible<unsigned, std::plus<unsigned> > ParCounter;

class AbstractWavefrontBFS: public BFS<Graph, GNode> {

protected:
  typedef BFS<Graph, GNode> SuperTy;

  //! @return number of iterations
  template <typename WL, typename WFInnerLoop> 
  static size_t runWavefrontBFS (Graph& graph, GNode& startNode, 
      void (WL::*pushFn) (const typename WL::value_type&),
      const WFInnerLoop& innerLoop) {

    WL* currWL = new WL ();
    WL* nextWL = new WL ();

    graph.getData (startNode, Galois::NONE) = 0;
    (currWL->*pushFn) (startNode);
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
  typedef std::vector<GNode> WL_ty;
  typedef AbstractWavefrontBFS::SuperTy BaseBFS;


private:
  struct SerialInnerLoop {
    unsigned operator () (Graph& graph, WL_ty& currWL, WL_ty& nextWL) const {

      unsigned numAdds = 0;

      for (WL_ty::iterator src = currWL.begin (), esrc = currWL.end ();
          src != esrc; ++src) {

        numAdds += BaseBFS::bfsOperator<false> (graph, *src, nextWL, &WL_ty::push_back);
      }

      return numAdds;
    }
  };



public:
  virtual const std::string getVersion () const { return "Serial Wavefront"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {
    return AbstractWavefrontBFS::runWavefrontBFS<WL_ty> (graph, startNode, &WL_ty::push_back, SerialInnerLoop ());
  }

};


template <bool doLock> 
struct LoopFlags {
  typedef int tt_does_not_need_stats; // disable stats in GaloisRuntime
  typedef int tt_does_not_need_parallel_push;
};

template <>
struct LoopFlags<false> { // more when no locking
  typedef int tt_does_not_need_stats; // disable stats in GaloisRuntime
  typedef char tt_does_not_need_parallel_push;
  typedef double tt_does_not_need_aborts;
};


class BFSwavefrontNolock;

class BFSwavefrontLock: public AbstractWavefrontBFS {
protected:
  static const unsigned CHUNK_SIZE = 1024; 
  typedef GaloisRuntime::WorkList::dChunkedFIFO<CHUNK_SIZE, GNode> GaloisWL;

  typedef AbstractWavefrontBFS::SuperTy BaseBFS;

  friend class BFSwavefrontNolock;

private:


  template <bool doLock>
  struct LoopBody: public LoopFlags<doLock> {

    Graph& graph;
    GaloisWL& nextWL;
    ParCounter& numAdds;

    LoopBody (
        Graph& graph,
        GaloisWL& nextWL,
        ParCounter& numAdds)
      :
        graph (graph),
        nextWL (nextWL),
        numAdds (numAdds) 
    {} 

    template <typename ContextTy>
    void operator () (GNode src, ContextTy&) {
      numAdds.get () += BaseBFS::bfsOperator<doLock> (graph, src, nextWL, &GaloisWL::push);
    }
  };


  template <bool doLock>
  struct ParallelInnerLoop {
    unsigned operator () (Graph& graph, GaloisWL& currWL, GaloisWL& nextWL) const {

      ParCounter numAdds (0);

      LoopBody<doLock> l (graph, nextWL, numAdds);
      Galois::for_each_wl (currWL, l);
      // Galois::for_each_wl <GaloisRuntime::WorkList::ParaMeter<GaloisWL> > (currWL, l);

      return numAdds.reduce ();
    }

  };

public:
  virtual const std::string getVersion () const { return "Galois Wavefront with Locking"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {
    
    return AbstractWavefrontBFS::runWavefrontBFS<GaloisWL> (
        graph, startNode, &GaloisWL::push, 
        ParallelInnerLoop<true> ()); // true means acquire locks
  }

};

class BFSwavefrontNolock: public AbstractWavefrontBFS {
  virtual const std::string getVersion () const { return "Galois Wavefront NO Locking"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {
    
    return AbstractWavefrontBFS::runWavefrontBFS<BFSwavefrontLock::GaloisWL> (
        graph, startNode, 
        &BFSwavefrontLock::GaloisWL::push, 
        BFSwavefrontLock::ParallelInnerLoop<false> ()); // false for no locking
  }
};




class BFSwavefrontCoupled: public BFS<Graph, GNode> {

  typedef BFS<Graph, GNode> SuperTy;
  typedef GaloisRuntime::PerThreadWLfactory<GNode>::PerThreadVector WL_ty;

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

    void operator () (GNode src) {
      numAdds.get () += SuperTy::bfsOperator<false> (graph, src, nextWL.get (), &WL_ty::ContTy::push_back);
    }
  };

public:

  virtual const std::string getVersion () const { return "Galois Wavefront Coupled DoAll"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {

    WL_ty* currWL = new WL_ty ();
    WL_ty* nextWL = new WL_ty ();

    graph.getData (startNode, Galois::NONE) = 0;
    currWL->get ().push_back (startNode);

    size_t numIter = 1;

#ifdef GALOIS_USE_VTUNE
    __itt_resume ();
#endif

    ParCounter numAdds (0);
    while (!currWL->empty_all ()) {


      GaloisRuntime::do_all_coupled (*currWL, ParallelInnerLoop (graph, *nextWL, numAdds), "wavefront_inner_loop");

      std::swap (currWL, nextWL);
      nextWL->clear_all ();
    }

    numIter += numAdds.reduce ();

#ifdef GALOIS_USE_VTUNE
  __itt_pause ();
#endif

    delete currWL; currWL = NULL;
    delete nextWL; nextWL = NULL;

    return numIter;
  }

};




#endif // WAVEFRONT_BFS_H_
