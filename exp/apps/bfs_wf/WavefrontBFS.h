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
#include "Galois/Bag.h"

#include "Galois/Runtime/PerThreadWorkList.h"
#include "Galois/Runtime/DoAll.h"

#include "bfs.h"

static const unsigned CHUNK_SIZE = 1024; 

typedef Galois::GAccumulator<unsigned> ParCounter;

class AbstractWavefrontBFS: public BFS<NodeData> {

protected:
  typedef BFS<NodeData> Super_ty;

  //! @return number of iterations
  template <typename WL, typename WFInnerLoop> 
  static size_t runWavefrontBFS (
      Graph& graph, 
      GNode& startNode, 
      void (WL::*pushFn) (const typename WL::value_type&),
      const WFInnerLoop& innerLoop) {

    WL* currWL = new WL ();
    WL* nextWL = new WL ();

    graph.getData (startNode, Galois::NONE).level () = 0;
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
  typedef AbstractWavefrontBFS::Super_ty Super_ty;


private:
  struct SerialInnerLoop {
    GALOIS_ATTRIBUTE_PROF_NOINLINE unsigned operator () (Graph& graph, WL_ty& currWL, WL_ty& nextWL) const {

      unsigned numAdds = 0;

      for (WL_ty::iterator src = currWL.begin (), esrc = currWL.end ();
          src != esrc; ++src) {

        numAdds += Super_ty::bfsOperator<false> (graph, *src, nextWL, &WL_ty::push_back);
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


template <typename WL, typename ND> 
struct InnerLoopDoAll {
  typedef BFS<ND> BaseBFS;

  typename BaseBFS::Graph& graph;
  WL& nextWL;
  ParCounter& numAdds;

  InnerLoopDoAll (
      typename BaseBFS::Graph& _graph,
      WL& _nextWL,
      ParCounter& _numAdds)
    :
      graph (_graph),
      nextWL (_nextWL),
      numAdds (_numAdds)
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (typename BaseBFS::GNode& src) {
    numAdds.get () += BFS<ND>::template bfsOperator<false, WL> (graph, src, nextWL, &WL::push);
  }

};

template <bool doLock> 
struct LoopFlags {
  typedef int tt_does_not_need_stats; // disable stats in Galois::Runtime
  typedef int tt_does_not_need_push;
};

template <>
struct LoopFlags<false> { // more when no locking
  typedef int tt_does_not_need_stats; // disable stats in Galois::Runtime
  typedef char tt_does_not_need_push;
  typedef double tt_does_not_need_aborts;
};


template <bool doLock, typename WL, typename ND>
struct InnerLoopForEach: public InnerLoopDoAll<WL, ND>, public LoopFlags<doLock> {

  typedef InnerLoopDoAll<WL, ND> Super_ty;
  typedef BFS<ND> BaseBFS;

  InnerLoopForEach (
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

class BFSwavefrontNolock;

class BFSwavefrontLock: public AbstractWavefrontBFS {
protected:
  typedef Galois::Runtime::WorkList::dChunkedFIFO<CHUNK_SIZE, GNode> GaloisWL;

  typedef AbstractWavefrontBFS::Super_ty BaseBFS;

  friend class BFSwavefrontNolock;

private:




  template <bool doLock>
  struct ParallelInnerLoop {
    GALOIS_ATTRIBUTE_PROF_NOINLINE unsigned operator () (Graph& graph, GaloisWL& currWL, GaloisWL& nextWL) const {

      ParCounter numAdds;

      InnerLoopForEach<doLock, GaloisWL, Super_ty::NodeData_ty> l (graph, nextWL, numAdds);
      Galois::for_each_wl (currWL, l);
      // Galois::for_each_wl <Galois::Runtime::WorkList::ParaMeter<GaloisWL> > (currWL, l);

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

class BFSwavefrontBag: public AbstractWavefrontBFS {

  template <typename T>
  struct BFSbag: public Galois::InsertBag<T> {
    void push (const T& val) {
      Galois::InsertBag<T>::push (val);
    }
  };
  typedef BFSbag<GNode> WL_ty;

#if 0
  template <typename T>
  struct BFSbag: public Galois::MergeBag<T> {
    void push (const T& v) {
      Galois::MergeBag<T>::push_back (v);
    }
  };
  typedef BFSbag<GNode> WL_ty;
#endif

  struct ParallelInnerLoop {
    GALOIS_ATTRIBUTE_PROF_NOINLINE unsigned operator () (Graph& graph, WL_ty& currWL, WL_ty& nextWL) const {

      ParCounter numAdds;

      InnerLoopDoAll<WL_ty, Super_ty::NodeData_ty> l (graph, nextWL, numAdds);

      Galois::do_all (currWL.begin (), currWL.end (), l, "bag-based do-all");
      // Galois::Runtime::do_all_coupled (currWL.begin (), currWL.end (), l, "bag-based do-all", CHUNK_SIZE);

      nextWL.merge ();

      return numAdds.reduce ();
    }

  };


  virtual const std::string getVersion () const { return "Galois Wavefront bag-based do-all"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {
    
    return AbstractWavefrontBFS::runWavefrontBFS<WL_ty> (
        graph, startNode, 
        &WL_ty::push, 
        ParallelInnerLoop ()); // false for no locking
  }
};



class BFSwavefrontCoupled: public AbstractWavefrontBFS {

  // typedef Galois::Runtime::PerThreadWLfactory<GNode>::PerThreadVector WL_ty;
  typedef Galois::Runtime::PerThreadVector<GNode> WL_ty;

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

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (GNode src) {
      typedef WL_ty::Cont_ty C;
      numAdds.get () += Super_ty::bfsOperator<false> (graph, src, nextWL.get (), &C::push_back);
    }
  };

public:

  virtual const std::string getVersion () const { return "Galois Wavefront Coupled DoAll"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {

    WL_ty* currWL = new WL_ty ();
    WL_ty* nextWL = new WL_ty ();

    graph.getData (startNode, Galois::NONE).level () = 0;
    currWL->get ().push_back (startNode);

    size_t numIter = 1;

    unsigned level = 0;

    ParCounter numAdds;
    // while (!currWL->empty_all ()) {
    for (size_t s = currWL->size_all (); s != 0; s = currWL->size_all ()) {

      // // TODO: remove
      // if (level == 4) {
        // Galois::Runtime::beginSampling ();
      // }

      size_t chunk_size = std::max (size_t(1), s/ (16 * Galois::getActiveThreads ()));

      // Galois::Runtime::do_all_coupled (*currWL, ParallelInnerLoop (graph, *nextWL, numAdds), "wavefront_inner_loop", chunk_size);
      // Galois::Runtime::do_all_coupled (currWL->begin_all (), currWL->end_all (), ParallelInnerLoop (graph, *nextWL, numAdds), "wavefront_inner_loop", chunk_size);
      Galois::do_all (currWL->begin_all (), currWL->end_all (), ParallelInnerLoop (graph, *nextWL, numAdds), "wavefront_inner_loop");

      // TODO: remove
      // if (level == 4) {
        // Galois::Runtime::endSampling ();
      // }

      std::swap (currWL, nextWL);
      nextWL->clear_all ();
      ++level;
    }

    numIter += numAdds.reduce ();

    delete currWL; currWL = NULL;
    delete nextWL; nextWL = NULL;

    return numIter;
  }

};

class BFSwavefrontEdge: public AbstractWavefrontBFS {


  struct Update {
    GNode node;
    unsigned dist;

    Update (GNode _node, unsigned _dist):
      node (_node), dist (_dist) {}
  };

  typedef Galois::Runtime::PerThreadDeque<Update> WL_ty;

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

    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (Update& up) {
      GNode src = up.node;
      const unsigned newLevel = up.dist + 1; // src level + 1

      for (Graph::edge_iterator ni = graph.edge_begin (src, Galois::NONE), eni = graph.edge_end (src, Galois::NONE);
          ni != eni; ++ni) {

        GNode dst = graph.getEdgeDst (ni);

        Super_ty::NodeData_ty& dstData = graph.getData (dst, Galois::NONE);

        if (dstData.level () > newLevel) {
          dstData.level () = newLevel;
          // nextWL.get ().push_back (Update (dst, newLevel));
          Super_ty::addToWL (
              nextWL.get (), 
              &WL_ty::Cont_ty::push_back, 
              Update (dst, newLevel));
          ++(numAdds.get ());
        }

      }

    }
  };

public:

  virtual const std::string getVersion () const { return "Galois Wavefront DoAll Edge based version"; }

  virtual size_t runBFS (Graph& graph, GNode& startNode) {

    WL_ty* currWL = new WL_ty ();
    WL_ty* nextWL = new WL_ty ();


    graph.getData (startNode, Galois::NONE).level () = 0;
    currWL->get ().push_back (Update (startNode, 0));

    size_t numIter = 1;

    unsigned level = 0;

    ParCounter numAdds;
    while (!currWL->empty_all ()) {

      Galois::Runtime::do_all_coupled (*currWL, ParallelInnerLoop (graph, *nextWL, numAdds), "wavefront_inner_loop", CHUNK_SIZE);
      // Galois::Runtime::do_all_coupled_reverse (*currWL, ParallelInnerLoop (graph, *nextWL, numAdds), "wavefront_inner_loop", CHUNK_SIZE);


      std::swap (currWL, nextWL);
      nextWL->clear_all ();
      ++level;
    }

    numIter += numAdds.reduce ();

    delete currWL; currWL = NULL;
    delete nextWL; nextWL = NULL;

    return numIter;
  }

};



#endif // WAVEFRONT_BFS_H_

