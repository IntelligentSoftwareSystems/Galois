/** Kruskal parallel but without keeping adj lists -*- C++ -*-
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
 * No adjacency lists per component. Adj information (min outgoing edge)
 * is recomputed when executing
 *
 * @author <ahassaan@ices.utexas.edu>
 */

#ifndef _KRUSKAL_NO_ADJ_H_
#define _KRUSKAL_NO_ADJ_H_

#include <algorithm>

#include <cstdio>

#include <boost/iterator/counting_iterator.hpp>

#include "galois/Timer.h"
#include "galois/Accumulator.h"
#include "galois/PerThreadContainer.h"
#include "galois/util/Markable.h"

#include "galois/runtime/DoAllCoupled.h"
#include "galois/runtime/ll/CompilerSpecific.h"



#include "kruskalData.h"
#include "kruskalFunc.h"
#include "kruskal.h"


typedef galois::GAccumulator<size_t> Accumulator_ty;


template <typename KNode_tp>
struct WLfactory {

  typedef KEdge<KNode_tp> Edge_ty;
  typedef typename galois::PerThreadVector<Edge_ty* > WL_ty;
  typedef typename galois::PerThreadVector<Markable<Edge_ty* > > WLmarkable_ty;

};


template <typename KNode_tp>
struct FindLoop {

  Accumulator_ty& matchIter;

  FindLoop (
      Accumulator_ty& _matchIter)
    :
      matchIter (_matchIter)
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE static void updateODG_claim (KEdge<KNode_tp>* edge, KNode_tp* rep1, KNode_tp* rep2) {
    if (rep1 != rep2) {
      rep1->claimAsMin (edge);
      rep2->claimAsMin (edge);
    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (KEdge<KNode_tp>* edge) {
    matchIter += 1;

    KNode_tp* rep1 = kruskal::findPC (edge->src);

    KNode_tp* rep2 = kruskal::findPC (edge->dst);

    updateODG_claim (edge, rep1, rep2);
  }
};

template <typename KNode_tp>
struct LinkUpLoop {

  typedef typename WLfactory<KNode_tp>::WL_ty WL_ty;

  WL_ty& nextWorkList;
  unsigned round;
  Accumulator_ty& numUnions;
  Accumulator_ty& mstSum;
  Accumulator_ty& mergeIter;

  LinkUpLoop (
      WL_ty& _nextWorkList,
      unsigned _round,
      Accumulator_ty& _numUnions,
      Accumulator_ty& _mstSum,
      Accumulator_ty& _mergeIter)
    :
      nextWorkList (_nextWorkList),
      round (_round),
      numUnions (_numUnions),
      mstSum (_mstSum),
      mergeIter (_mergeIter)
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE static bool updateODG_test (KEdge<KNode_tp>& edge, KNode_tp* rep) {
    assert (rep != NULL);
    return (rep->minEdge == &edge);
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE static void updateODG_reset (KEdge<KNode_tp>& edge, KNode_tp* rep) {
    if (updateODG_test (edge, rep)) {
      rep->minEdge = NULL;
    }
  }

  GALOIS_ATTRIBUTE_PROF_NOINLINE static void addToWL (WL_ty& wl, const KEdge<KNode_tp>& edge) {
    wl.get ().push_back (const_cast<KEdge<KNode_tp>*> (&edge));
  }


  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (KEdge<KNode_tp>* e) {

    assert (e != NULL);
    KEdge<KNode_tp>& edge = *e;


    // relies on find with path-compression
    KNode_tp* rep1 = edge.src->getRep ();
    KNode_tp* rep2 = edge.dst->getRep ();


    // not  a self-edge
    if (rep1 != rep2) {
      mergeIter += 1;

      bool succ1 = updateODG_test (edge, rep1);
      bool succ2 = updateODG_test (edge, rep2);

      if (succ1) {
        kruskal::linkUp (rep1, rep2);

      } else if (succ2) {
        kruskal::linkUp (rep2, rep1);

      } else {
        // nextWorkList.get ().push_back (&edge);
        addToWL (nextWorkList, edge);
      }


      if (succ1 || succ2) {
        numUnions += 1;
        mstSum += edge.weight;
        edge.inMST = true;

        // reset minEdge for next round
        // only on success
        // if (succ1) {
          // rep1->minEdge = NULL;
        // }

        // if (succ2) {
          // rep2->minEdge = NULL;
        // }
        updateODG_reset (edge, rep1);
        updateODG_reset (edge, rep2);
      }
    }

  }

};


template <typename KNode_tp>
void kruskalNoAdjNonSrc (
    std::vector<KNode_tp*>& nodes, 
    std::vector<KEdge<KNode_tp>* >& edges, 
    size_t& totalWeight, 
    size_t& totalIter) {

  // fill in a per thread currWorkList of marked edges and sort (for efficiency);
  // loop where edges compete for lightest using Find . Mark self edges
  // loop where edges check if they won and merge using union
  // loop to remove the marked edges

  totalIter = 0;
  totalWeight = 0;

  unsigned round = 0;
  size_t totalUnions = 0;


  Accumulator_ty matchIter;
  Accumulator_ty mstSum;
  Accumulator_ty mergeIter;
  Accumulator_ty numUnions;

  galois::TimeAccumulator matchTimer;
  galois::TimeAccumulator mergeTimer;


  typedef typename WLfactory<KNode_tp>::WL_ty WL_ty;


  WL_ty* currWorkList = new WL_ty ();
  WL_ty* nextWorkList = new WL_ty ();

  bool first = true;

  while (true) {

    if (first) {
      first = false;

      matchTimer.start ();

      galois::runtime::do_all_coupled (
          edges.begin (), edges.end (),
          FindLoop<KNode_tp> (matchIter), "match_loop");

      matchTimer.stop ();


      mergeTimer.start ();

      galois::runtime::do_all_coupled (
          edges.begin (), edges.end (),
          LinkUpLoop<KNode_tp> (*nextWorkList, round, numUnions, mstSum, mergeIter), "merge_loop");

      mergeTimer.stop ();

    } else {

      matchTimer.start ();

      galois::runtime::do_all_coupled (
          *currWorkList,
          FindLoop<KNode_tp> (matchIter), "match_loop");

      matchTimer.stop ();


      mergeTimer.start ();

      galois::runtime::do_all_coupled (
          *currWorkList,
          LinkUpLoop<KNode_tp> (*nextWorkList, round, numUnions, mstSum, mergeIter), "merge_loop");

      mergeTimer.stop ();
    }

    size_t u = numUnions.reduce () - totalUnions;
    assert (u > 0 && "No merges in this round, no progress???");
    
    totalUnions += u;


    if (totalUnions == (nodes.size () - 1)) {
      break;
    }


    ++round;
    std::swap (currWorkList, nextWorkList);
    nextWorkList->clear_all ();
    
  }


  totalWeight = mstSum.reduce ();
  totalIter = matchIter.reduce ();


  std::cout << "Number of match-merge rounds: " << round << std::endl;

  std::cout << "FindLoop iterations = " << matchIter.reduce () << std::endl;
  std::cout << "LinkUpLoop iterations = " << mergeIter.reduce () << std::endl;
  std::cout << "Match to Merge iteration ratio = " 
    << (double (matchIter.reduce ()) / numUnions.reduce ()) << std::endl;

  std::cout << "Total time taken by FindLoop: " << matchTimer.get () << std::endl;
  std::cout << "Total time taken by LinkUpLoop : " << mergeTimer.get () << std::endl;

  delete currWorkList; currWorkList = NULL;
  delete nextWorkList; nextWorkList = NULL;

}




// low parallelism threshold, at which kruskal switches to serial
static const size_t LOW_PAR_THRESH = 128;

template <typename KNode_tp>
struct FindLoopMarked: public FindLoop<KNode_tp> {

  FindLoopMarked (Accumulator_ty& _matchIter): FindLoop<KNode_tp> (_matchIter) {}
  
  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (Markable<KEdge<KNode_tp> >& edge) {
    if (!edge.marked ()) {
      FindLoop<KNode_tp>::operator () (edge);
    }
  }

};


template <typename KNode_tp>
struct UnionLoop {

  unsigned round;
  Accumulator_ty& numUnions;
  Accumulator_ty& mstSum;
  Accumulator_ty& mergeIter;

  UnionLoop (
      unsigned _round,
      Accumulator_ty& _numUnions,
      Accumulator_ty& _mstSum,
      Accumulator_ty& _mergeIter)
    :
      round (_round),
      numUnions (_numUnions),
      mstSum (_mstSum),
      mergeIter (_mergeIter)
  {}

  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (Markable<KEdge<KNode_tp>* >& medge) {

    KEdge<KNode_tp>& edge = *medge;


    KNode_tp* rep1 = edge.src->getRep ();
    KNode_tp* rep2 = edge.dst->getRep ();

    if (!medge.marked () && (rep1 != rep2)) {
      mergeIter += 1;


      bool succ1 = (rep1->minEdge == &edge);
      bool succ2 = (rep2->minEdge == &edge);

      if (succ1 && succ2) {

        kruskal::unionByRank (rep1, rep2);


        numUnions += 1;
        mstSum += edge.weight;
        edge.inMST = true;

        // mark the edge
        medge.mark (round);

        // reset minEdge for next round
        if (succ1) {
          rep1->minEdge = NULL;
        }

        if (succ2) {
          rep2->minEdge = NULL;
        }
      }

    } else {
      if (!medge.marked ()) {
        // a self edge, not marked
        medge.mark (round);
      }
    }
  }
};


template <typename KNode_tp>
void kruskalNoAdjSrc (std::vector<KNode_tp*>& nodes, 
    std::vector<KEdge<KNode_tp>* >& edges, 
    size_t& totalWeight, size_t& totalIter) {

  // fill in a per thread workList of marked edges and sort (for efficiency);
  // loop where edges compete for lightest using Find . Mark self edges
  // loop where edges check if they won and merge using union
  // loop to remove the marked edges

  const bool ENABLE_SERIAL = true;

  if (!ENABLE_SERIAL) {
    std::cerr << "WARNING: switching to serial disabled (kruskalNoAdjSrc())" << std::endl;
  }

  totalIter = 0;
  totalWeight = 0;

  unsigned round = 0;
  size_t totalUnions = 0;


  Accumulator_ty matchIter;
  Accumulator_ty mstSum;
  Accumulator_ty mergeIter;
  Accumulator_ty numUnions;

  galois::TimeAccumulator matchTimer;
  galois::TimeAccumulator mergeTimer;
  galois::TimeAccumulator removeTimer;

  typedef typename WLfactory<KNode_tp>::WLmarkable_ty WL_ty;


  WL_ty workList;

  galois::StatTimer t_wl_init ("WorkList initialization time: ");

  t_wl_init.start ();
  workList.fill_init (edges.begin (), edges.end (), &WL_ty::Cont_ty::push_back);
  t_wl_init.stop ();



  
  while (true) {


    matchTimer.start ();

    galois::runtime::do_all_coupled (
        workList,
        FindLoop<KNode_tp> (matchIter), "match_loop");

    matchTimer.stop ();


    mergeTimer.start ();

    // std::cout << "Starting merge_loop" << std::endl;

    galois::runtime::do_all_coupled (
        workList,
        UnionLoop<KNode_tp> (round, numUnions, mstSum, mergeIter), "merge_loop");

    mergeTimer.stop ();

    size_t u = numUnions.reduce () - totalUnions;
    assert (u > 0);
    
    totalUnions += u;


    if (totalUnions == (nodes.size () - 1)) {
      break;
    }

    removeTimer.start ();
    removeMarked (workList);
    removeTimer.stop ();


    if (ENABLE_SERIAL && (u < LOW_PAR_THRESH)) {
      break;
    }


    ++round;
    
  }


  totalWeight = mstSum.reduce ();
  totalIter = matchIter.reduce ();


  if (totalUnions < (nodes.size () - 1)) {

    galois::StatTimer t_serial ("Time for serially processing remainging edges");

    t_serial.start ();

    typedef std::vector<Markable<KEdge<KNode_tp>* > > SWL_ty;

    SWL_ty serialWorkList;

    for (unsigned r = 0; r < workList.numRows (); ++r) {
      serialWorkList.insert (
          serialWorkList.end (), 
          workList[r].begin (), workList[r].end ());
    }

    std::sort (serialWorkList.begin (), serialWorkList.end (), 
        typename KEdge<KNode_tp>::PtrComparator ());


    size_t niter = 0;
    for (typename SWL_ty::iterator i = serialWorkList.begin (), ei = serialWorkList.end ();
        i != ei; ++i) {

      ++niter;

      assert (!i->marked ());

      if (kruskal::contract (**i)) {
        ++totalUnions;
        totalWeight += i->get ()->weight;
      }

      if (totalUnions == (nodes.size () - 1)) {
        break;
      }
    }

    t_serial.stop ();

    std::cout << "Number of edges processed serially = " << niter << std::endl;

    totalIter += niter;

    // final check. no non-self edges remaining
    for (typename SWL_ty::iterator i = serialWorkList.begin (), ei = serialWorkList.end (); 
        i != ei; ++i) {

      assert (!kruskal::NotSelfEdge<KNode_tp> () (*i));
    }
  }




  std::cout << "Number of match-merge rounds: " << round << std::endl;

  std::cout << "FindLoop iterations = " << matchIter.reduce () << std::endl;
  std::cout << "UnionLoop iterations = " << mergeIter.reduce () << std::endl;



  std::cout << "Total time taken by FindLoop: " << matchTimer.get () << std::endl;
  std::cout << "Total time taken by UnionLoop : " << mergeTimer.get () << std::endl;
  std::cout << "Total time taken by RemoveMarked : " << removeTimer.get () << std::endl;



}

class KruskalNoAdjNonSrc: public Kruskal<KNodeMin> {
protected:

  virtual const std::string getVersion () const { return "Kruskal no-adj sources, and, non-sources satisfying Boruvka Property selected"; }

  virtual void runMST (Base_ty::VecKNode_ty& nodes, Base_ty::VecKEdge_ty& edges,
      size_t& totalWeight, size_t& totalIter) {

    kruskalNoAdjNonSrc (nodes, edges, totalWeight, totalIter);
  }
};

class KruskalNoAdjSrc: public Kruskal<KNodeMin> {
protected:

  virtual const std::string getVersion () const { return "Kruskal no-adj, only sources selected"; }

  virtual void runMST (Base_ty::VecKNode_ty& nodes, Base_ty::VecKEdge_ty& edges,
      size_t& totalWeight, size_t& totalIter) {

    kruskalNoAdjSrc (nodes, edges, totalWeight, totalIter);
  }
};




#endif //  _KRUSKAL_NO_ADJ_H_


