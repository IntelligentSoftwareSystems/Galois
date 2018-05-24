#ifndef _KRUSKAL_ADJ_H_
#define _KRUSKAL_ADJ_H_

#include <algorithm>
#include <iostream>
#include <fstream>

#include "galois/Atomic.h"
#include "galois/Timer.h"
#include "galois/util/Markable.h"
#include "galois/Reduction.h"
#include "galois/PerThreadContainer.h"

#include "galois/runtime/DoAllCoupled.h"


#include "kruskalData.h"
#include "kruskalFunc.h"
#include "kruskal.h"



typedef galois::GAccumulator<size_t> Accumulator_ty;


template <typename KNode_tp>
struct WLfactory {
  typedef KEdge<KNode_tp> KE_ty;

  typedef typename galois::PerThreadVector<KE_ty*> EdgeList_ty;
  typedef typename galois::PerThreadVector<KNode_tp*> NodeList_ty;

  typedef typename galois::PerThreadVector<Markable<KE_ty*> > MarkableEdgeList_ty;
};


template <bool sorted_tp>
void arrangeWrapper (KNodeAdj* node) {
  kruskal::arrangeAdj<sorted_tp> (*node);
}



  
template <bool sorted_tp, typename Flavor_tp, typename KNode_tp>
void kruskalAdjNoCopy (
    std::vector<KNode_tp*>& nodes,
    std::vector<KEdge<KNode_tp>* >& edges,
    size_t& totalWeight,
    size_t& totalIter) {

  // create marked list of nodes or edges
  // run match loop on marked items
  // run merge loop on edges
  // if num marked is above threshold, remove marked items
  //

  typedef typename Flavor_tp::MatchListTy MatchListTy;
  typedef typename Flavor_tp::MergeListTy MergeListTy;
  typedef typename Flavor_tp::MatchLoopTy MatchLoopTy;
  typedef typename Flavor_tp::MergeLoopTy MergeLoopTy;

  galois::StatTimer t_arrange ("Time taken to arrange adjacency lists: ");

  t_arrange.start ();
  galois::runtime::do_all_coupled (nodes.begin (), nodes.end (), &arrangeWrapper<sorted_tp>, "adj_pre_arrange_loop");
  t_arrange.stop ();


  MatchListTy matchList;
  MergeListTy mergeList;



  Flavor_tp flavor (nodes, edges);

  galois::StatTimer t_init_fill ("Time spent in initializing per thread work-list: ");
  t_init_fill.start ();
  flavor.fillInitial (nodes, edges, matchList);
  t_init_fill.stop ();



  totalIter = 0;
  totalWeight = 0;
  size_t numUnions = 0;
  unsigned round = 0;

  Accumulator_ty matchIter;
  Accumulator_ty mstSum;
  Accumulator_ty mergeIter;

  galois::TimeAccumulator matchTimer;
  galois::TimeAccumulator mergeTimer;
  galois::TimeAccumulator removeTimer;


  while (true) {
    ++round;

    matchTimer.start ();
    galois::runtime::do_all_coupled (
        matchList, 
        flavor.makeMatchLoop (mergeList, matchIter),
        "match_loop");
    matchTimer.stop ();


    mergeTimer.start ();
    galois::runtime::do_all_coupled (
        mergeList,
        flavor.makeMergeLoop (mstSum, mergeIter),
        "merge_loop");
    mergeTimer.stop ();

    size_t u = mergeIter.reduce () - numUnions;

    assert ((u != 0) && "No merges, no progress?");
    numUnions += u;

    if (numUnions == (nodes.size () - 1)) {
      break;
    }

    // TODO: we'd like to switch to serial when parallelism i.e. u
    // drops below some threshold

    if (true) { // TODO: Condition of when to clean up goes here

      removeTimer.start ();
      removeMarked (matchList);
      removeTimer.stop ();

    }

    mergeList.clear_all ();
  }


  totalWeight = mstSum.reduce ();
  totalIter = matchIter.reduce ();
  
  std::cout << "Number of match-merge rounds: " << round << std::endl;

  std::cout << "FindLoop iterations = " << matchIter.reduce () << std::endl;
  std::cout << "UnionLoop iterations = " << mergeIter.reduce () << std::endl;



  std::cout << "Total time taken by FindLoop: " << matchTimer.get () << std::endl;
  std::cout << "Total time taken by UnionLoop : " << mergeTimer.get () << std::endl;
  std::cout << "Total time taken by RemoveMarked : " << removeTimer.get () << std::endl;


}



template <bool sorted_tp, typename KNode_tp> 
struct KruskalAdjEdgeBased {

  typedef typename WLfactory<KNode_tp>::MarkableEdgeList_ty MatchListTy;
  typedef typename WLfactory<KNode_tp>::MarkableEdgeList_ty MergeListTy ;

private:

 
  struct MatchLoopEdgeBased {
    MergeListTy& mergeList;
    Accumulator_ty& matchIter;

    MatchLoopEdgeBased (
        MergeListTy& mergeList,
        Accumulator_ty& matchIter)
      :
        mergeList (mergeList),
        matchIter (matchIter) 
    {}


    void operator () (Markable<KEdge<KNode_tp>* >& medge) const {
      matchIter += 1;

      KEdge<KNode_tp>* edge = medge;

      if (!medge.marked () && kruskal::NotSelfEdge<KNode_tp> () (edge)) {

        KNode_tp* rep1 = kruskal::findPC (edge->src);
        KNode_tp* rep2 = kruskal::findPC (edge->dst);

        assert (rep1 != rep2);

        if (kruskal::getLightest<sorted_tp> (*rep1) == edge &&
            kruskal::getLightest<sorted_tp> (*rep2) == edge) {

          mergeList.get ().push_back (medge);

          medge.mark (0);

        } 
      } else {
        medge.mark (0);
      }

    }


  };


public:

  struct MergeLoopEdgeBased {

    Accumulator_ty& mstSum;
    Accumulator_ty& mergeIter;

    MergeLoopEdgeBased (
        Accumulator_ty& mstSum,
        Accumulator_ty& mergeIter)
      :
        mstSum (mstSum),
        mergeIter (mergeIter) 
    {}


      void operator () (Markable<KEdge<KNode_tp>* >& edge) const {
        doUnion (edge);
      }

    protected:
    std::pair<KNode_tp*, KNode_tp*> doUnion (KEdge<KNode_tp>* edge) const {

      assert (kruskal::NotSelfEdge<KNode_tp> () (edge));
      assert (!edge->inMST);

      std::pair<KNode_tp*, KNode_tp*> p = 
        kruskal::unionByRank (edge->src->getRep (), edge->dst->getRep ());

      KNode_tp& master = *(p.first);
      KNode_tp& other = *(p.second);

      kruskal::merge<sorted_tp> (master, other);

      edge->inMST = true;
      mstSum += edge->weight;
      mergeIter += 1;

      return p;
    }

  };


  typedef MatchLoopEdgeBased MatchLoopTy;
  typedef MergeLoopEdgeBased MergeLoopTy;

  KruskalAdjEdgeBased (
      const std::vector<KNode_tp*>& nodes,
      const std::vector<KEdge<KNode_tp>* >& edges) {
  }
      

  void fillInitial (
      const std::vector<KNode_tp*>& nodes,
      const std::vector<KEdge<KNode_tp>* >& edges,
      MatchListTy& matchList) const {

    matchList.fill_init (edges.begin (), edges.end (), &MatchListTy::Cont_ty::push_back);
  }

  MatchLoopTy makeMatchLoop (
      MergeListTy& mergeList,
      Accumulator_ty& matchIter) const {

    return MatchLoopTy (mergeList, matchIter);
  }
  
  MergeLoopTy makeMergeLoop (
      Accumulator_ty& mstSum,
      Accumulator_ty& mergeIter) const {

    return MergeLoopTy (mstSum, mergeIter);
  }
  
  void finishRound () const {}
};



template <bool sorted_tp, typename Flavor_tp, typename KNode_tp>
void kruskalAdjCopyBased (
    std::vector<KNode_tp*>& nodes,
    std::vector<KEdge<KNode_tp>* >& edges,
    size_t& totalWeight,
    size_t& totalIter) {


  typedef typename Flavor_tp::MatchListTy MatchListTy;
  typedef typename Flavor_tp::MergeListTy MergeListTy;
  typedef typename Flavor_tp::MatchLoopTy MatchLoopTy;
  typedef typename Flavor_tp::MergeLoopTy MergeLoopTy;


  galois::StatTimer t_arrange ("Time taken to arrange adjacency lists: ");

  t_arrange.start ();
  // for (std::vector<KNode_tp*>::iterator i = nodes.begin (), ei = nodes.end ();
      // i != ei; ++i) {
// 
    // kruskal::arrangeAdj<sorted_tp> (**i);
// 
  // }
  galois::runtime::do_all_coupled (nodes.begin (), nodes.end (), &arrangeWrapper<sorted_tp>, "adj_pre_arrange_loop");
  t_arrange.stop ();



  MatchListTy* matchList = new MatchListTy ();
  MatchListTy* nextRoundList = new MatchListTy ();
  MergeListTy mergeList;


  Flavor_tp flavor (nodes, edges);

  // use nodes in the first round instead
  // flavor.fillInitial (nodes, edges, *matchList);

  totalIter = 0;
  totalWeight = 0;
  size_t numUnions = 0;
  unsigned round = 0;


  Accumulator_ty matchIter;
  Accumulator_ty mstSum;
  Accumulator_ty mergeIter;


  galois::TimeAccumulator matchTimer;
  galois::TimeAccumulator mergeTimer;

  bool first = true;



  while (true) {
    ++round;

    matchTimer.start ();

    if (first) {
      first = false;
      galois::runtime::do_all_coupled (
          nodes.begin (), nodes.end (), 
          flavor.makeMatchLoop (*nextRoundList, mergeList, matchIter)
          , "match_loop");

    } else {

      galois::runtime::do_all_coupled (
          *matchList, 
          flavor.makeMatchLoop (*nextRoundList, mergeList, matchIter), 
          "match_loop");
    }

    matchTimer.stop ();


    mergeTimer.start ();

    galois::runtime::do_all_coupled (mergeList, 
        flavor.makeMergeLoop (*nextRoundList, mstSum, mergeIter), 
        "merge_loop");

    mergeTimer.stop ();

    unsigned u = mergeIter.reduce () - numUnions;
    assert ((u != 0) && "No merges? No progress?");
    numUnions += u;


    // std::cout << "Match Attemps:    " << matchIter.get () << std::endl;
    // std::cout << "Number of merges: " << mergeIter.get () << std::endl;
    // std::cout << "Max size of adjacency list recorded: " << kruskal::getMaxAdjSize () << std::endl;



    if (numUnions == (nodes.size () -1)) {
      break;
    }

    // switch workLists
    std::swap (matchList, nextRoundList);
    nextRoundList->clear_all ();
    assert (nextRoundList->empty_all ());

    // delete matchList;
    // matchList = nextRoundList;
    // nextRoundList = new MatchListTy();

    // empty mergeList
    mergeList.clear_all ();
    assert (mergeList.empty_all ());


    flavor.finishRound ();
    
  }


  totalIter = matchIter.reduce ();
  totalWeight = mstSum.reduce ();



  std::cout << "Number of match-merge rounds: " << round << std::endl;

  std::cout << "Max size of adjacency list recorded: " << kruskal::getMaxAdjSize () << std::endl;

  std::cout << "Total time taken by Match Loop: " << matchTimer.get () << std::endl;;
  std::cout << "Total time taken by Merge Loop: " << mergeTimer.get () << std::endl;;

  // after any k iterations matchList and nextRoundList point to 
  // valid freeable locations
  delete matchList;
  delete nextRoundList;

}



template <bool sorted_tp, typename KNode_tp>
struct KruskalAdjNodeBased {

  typedef typename WLfactory<KNode_tp>::NodeList_ty MatchListTy;
  typedef typename WLfactory<KNode_tp>::EdgeList_ty MergeListTy;

private:
  typedef KruskalAdjNodeBased<sorted_tp, KNode_tp> Outer;

  struct MatchLoopNodeBased {

    Outer& outer;
    MatchListTy& nextRoundList;
    MergeListTy& mergeList;
    Accumulator_ty& matchIter;

    MatchLoopNodeBased (
        Outer& outer,
        MatchListTy& nextRoundList,
        MergeListTy& mergeList,
        Accumulator_ty& matchIter)
      :
        outer (outer),
        nextRoundList (nextRoundList),
        mergeList (mergeList),
        matchIter (matchIter) 
    {}


    // TODO: fix comment
    // We start from a representative source node
    // and find its lightest non-self edge
    // then we find the representative of the
    // destination of this edge. If the edge
    // is lightest in the representative of the destination
    // node, then it's eligible for addition to the mergeList.
    // However, since both source and destination representatives
    // could be on the worklist, we need a mechanism
    // to avoid adding a lightest/source edge twice i.e. once processing source
    // representative and once processing the destination representative.
    // 
    // When both representatives are on the worklist,( e.g. in the beginning),
    // a possible solution is to compare the source and destination 
    // representatives on some criterion, where the winner of comparison adds the 
    // edge for merge phase, while loser skips. We keep a latestRound number for
    // each node, which is used as a criterion for selecting a winner. The winner
    // stays avlie, keeps updating its latestRound info, keeps winning unionByAge
    // 
    //
    void operator () (KNode_tp* node) const {

      matchIter += 1;

      // if is a representative
      if (kruskal::findPC (node) == node) {

        KEdge<KNode_tp>* edge = kruskal::getLightest<sorted_tp> (*node);

        KNode_tp* rep1 = kruskal::findPC (edge->src);
        KNode_tp* rep2 = kruskal::findPC (edge->dst);

        assert ((rep1 == node) || (rep2 == node));
        assert (rep1 != rep2);

        KNode_tp* otherRep = (rep1 == node) ? rep2: rep1;

        if (edge == kruskal::getLightest<sorted_tp> (*otherRep)) {

          if (outer.compareByAge (node, otherRep) > 0) {
            // node is going to win in union and will be added
            // to nextRoundList
            mergeList.get ().push_back (edge);
          } else {
            // other is going to win and should be on the current worklist
            // or else it's a bug
          }
        }
      } else {
        // drop because non-representative
      }

    }

  };


  // 
  bool compareByAge (KNode_tp* comp1, KNode_tp* comp2) {
    assert (comp1 != NULL);
    assert (comp2 != NULL);

    int res = latestRound[comp1->id] - latestRound[comp2->id];

    if (res == 0) {
      // tie break using ID when age is equal
      res = comp1->id - comp2->id;
    }

    assert (res != 0);
    return (res >= 0);

  }

  std::pair<KNode_tp*, KNode_tp*> unionByAge (KEdge<KNode_tp>* edge) {
    assert (kruskal::NotSelfEdge<KNode_tp> () (edge));
    assert (!edge->inMST);

    KNode_tp* rep1 = edge->src->getRep ();
    KNode_tp* rep2 = edge->dst->getRep ();

    KNode_tp* master = NULL;
    KNode_tp* other = NULL;

    if (compareByAge (rep1, rep2)) {
      master = rep1;
      other = rep2;

    } else {
      master = rep2;
      other = rep1;
    }

    kruskal::linkUp (other, master);
    latestRound[master->id] = round + 1;

    kruskal::merge<sorted_tp> (*master, *other);

    return std::make_pair (master, other);

  }

  struct MergeLoopNodeBased {


    Outer& outer;
    MatchListTy& nextRoundList;
    Accumulator_ty& mstSum;
    Accumulator_ty& mergeIter;

    MergeLoopNodeBased (
        Outer& outer,
        MatchListTy& nextRoundList,
        Accumulator_ty& mstSum,
        Accumulator_ty& mergeIter)
      :
        outer (outer),
        nextRoundList (nextRoundList),
        mstSum (mstSum),
        mergeIter (mergeIter)
    {}

    void operator () (KEdge<KNode_tp>* edge) {



      std::pair<KNode_tp*, KNode_tp*> p = outer.unionByAge (edge);

      edge->inMST = true;
      mstSum += edge->weight;
      mergeIter += 1;

      KNode_tp& master = *(p.first);

      if (!master.edges.empty ()) {
        // If all nodes are reachable from each other, 
        // this condition should only be true after the last union
        nextRoundList.get ().push_back (&master);
      }
    }

  };


  friend class MatchLoopNodeBased;
  friend class MergeLoopNodeBased;

  // the last time a node was involved in a merge
  std::vector<unsigned> latestRound;
  unsigned round;

public:

  typedef MatchLoopNodeBased MatchLoopTy;
  typedef MergeLoopNodeBased MergeLoopTy;

  KruskalAdjNodeBased (
      const std::vector<KNode_tp*>& nodes,
      const std::vector<KEdge<KNode_tp>* >& edges)
    :
      latestRound (nodes.size (), 0),
      round (0)
  {}



  MatchLoopTy makeMatchLoop (
      MatchListTy& nextRoundList,
      MergeListTy& mergeList,
      Accumulator_ty& matchIter) {

    return MatchLoopTy (*this, nextRoundList, mergeList, matchIter);
  }

  MergeLoopTy makeMergeLoop (
      MatchListTy& nextRoundList, 
      Accumulator_ty& mstSum,
      Accumulator_ty& mergeIter) {

    return MergeLoopTy (*this, nextRoundList, mstSum, mergeIter);
  }

  void finishRound () {
    ++round;
  }

};




template <bool sorted_tp, typename KNode_tp>
struct BoruvkaAdj {


  typedef typename WLfactory<KNode_tp>::NodeList_ty MatchListTy;
  typedef typename WLfactory<KNode_tp>::EdgeList_ty MergeListTy;

private:

  typedef BoruvkaAdj<sorted_tp, KNode_tp> Outer;

  struct MatchLoopBoruvka {

    Outer& outer;
    MatchListTy& nextRoundList;
    MergeListTy& mergeList;
    Accumulator_ty& matchIter;

    MatchLoopBoruvka (
        Outer& outer,
        MatchListTy& nextRoundList,
        MergeListTy& mergeList,
        Accumulator_ty& matchIter)
      :
        outer (outer),
        nextRoundList (nextRoundList),
        mergeList (mergeList),
        matchIter (matchIter) 
    {}


    
    void operator () (KNode_tp* node) {

      matchIter += 1;

      if ((kruskal::findPC (node) != node) ||
          (outer.matchInfo[node->id] != NULL)) {

        return;

      } else {

        KEdge<KNode_tp>* edge = kruskal::getLightest<sorted_tp> (*node);


        KNode_tp* rep1 = kruskal::findPC (edge->src);
        KNode_tp* rep2 = kruskal::findPC (edge->dst);

        assert ((rep1 == node) || (rep2 == node));
        assert (rep1 != rep2);

        KNode_tp* otherRep = (node == rep1)? rep2: rep1;

        if (didMatch (node, otherRep)) {
          mergeList.get ().push_back (edge);

        } else {
          nextRoundList.get ().push_back (node);
        }

      }

    }

  private:

    bool didMatch (KNode_tp* node, KNode_tp* otherRep) {
      bool success = false;

      if (outer.matchInfo[node->id].cas (NULL, otherRep)) {
        // matched node to otherRep successfully
        
        if (outer.matchInfo[otherRep->id].cas (NULL, node)) {
          // matched otherRep to node successfully
          success = true;

        } else {
          // could not match otherRep to node, so reset node's matchInfo
          outer.matchInfo[node->id] = NULL;
        }
      }

      return success;

    }


  };


  struct MergeLoopBoruvka : public KruskalAdjEdgeBased<sorted_tp, KNode_tp>::MergeLoopEdgeBased {

    typedef typename KruskalAdjEdgeBased<sorted_tp, KNode_tp>::MergeLoopEdgeBased SuperTy;

    Outer& outer;
    MatchListTy& nextRoundList;

    MergeLoopBoruvka (
        Outer& outer,
        MatchListTy& nextRoundList,
        Accumulator_ty& mstSum,
        Accumulator_ty& mergeIter)
      :
        SuperTy (mstSum, mergeIter),
        outer (outer),
        nextRoundList (nextRoundList)
    {}


    void operator () (KEdge<KNode_tp>* edge) const {

      std::pair<KNode_tp*, KNode_tp*> p = SuperTy::doUnion (edge);

      KNode_tp& master = *(p.first);
      KNode_tp& other = *(p.second);

      if (!master.edges.empty ()) {
        // If all nodes are reachable from each other, then
        // this condition should only be false after the last union
        nextRoundList.get ().push_back (&master);
      }

      outer.matchInfo [master.id] = NULL;
      outer.matchInfo [other.id] = NULL;
    }

  };


  friend class MatchLoopBoruvka;
  friend class MergeLoopBoruvka;

  typedef galois::GAtomic<KNode_tp*> MatchFlag;

  std::vector<MatchFlag> matchInfo;

public:

  typedef MatchLoopBoruvka MatchLoopTy;
  typedef MergeLoopBoruvka MergeLoopTy;

  BoruvkaAdj (
      const std::vector<KNode_tp*>& nodes,
      const std::vector<KEdge<KNode_tp>* >& edges)
    :
      matchInfo (nodes.size (), MatchFlag (NULL))
  {}



  MatchLoopTy makeMatchLoop (
      MatchListTy& nextRoundList,
      MergeListTy& mergeList,
      Accumulator_ty& matchIter) {

    return MatchLoopTy (*this, nextRoundList, mergeList, matchIter);
  }

  MergeLoopTy makeMergeLoop (
      MatchListTy& nextRoundList,
      Accumulator_ty& mstSum,
      Accumulator_ty& mergeIter) {

    return MergeLoopTy (*this, nextRoundList, mstSum, mergeIter);
  }

  void finishRound () const {}


};


//! this class initialize the adj lists of nodes properly
template <typename KNode_tp>
class KruskalAdjBase: public Kruskal<KNode_tp> {
  typedef Kruskal<KNode_tp> Super_ty;

protected:

  virtual void initRemaining (typename Super_ty::VecKNode_ty& nodes, 
      typename Super_ty::VecKEdge_ty& edges) {

    for (typename Super_ty::VecKEdge_ty::const_iterator e = edges.begin ()
        , ende = edges.end (); e != ende; ++e) {

      (*e)->src->addEdge (*e);
      (*e)->dst->addEdge (*e);

    }
  }
};

class KruskalAdjEdgeBasedSorted: public KruskalAdjBase<KNodeAdj> {
protected:

  virtual const std::string getVersion () const { return "Adjacency-based, Sorted, Match Loop is Edge-based"; }

  virtual void runMST (VecKNode_ty& nodes, VecKEdge_ty& edges,
      size_t& totalWeight, size_t& totalIter) {

    kruskalAdjNoCopy<true, KruskalAdjEdgeBased<true, KNodeAdj> > (nodes, edges, totalWeight, totalIter);
  }
};


class KruskalAdjEdgeBasedUnsorted: public KruskalAdjBase<KNodeAdj> {
protected:

  virtual const std::string getVersion () const { return "Adjacency-based, Unsorted, Match Loop is Edge-based"; }

  virtual void runMST (VecKNode_ty& nodes, VecKEdge_ty& edges,
      size_t& totalWeight, size_t& totalIter) {

    kruskalAdjNoCopy<false, KruskalAdjEdgeBased<false, KNodeAdj> > (nodes, edges, totalWeight, totalIter);
  }
};


class KruskalAdjNodeBasedSorted: public KruskalAdjBase<KNodeAdj> {
protected:

  virtual const std::string getVersion () const { return "Adjacency-based, Sorted, Match Loop is Node-based"; }

  virtual void runMST (VecKNode_ty& nodes, VecKEdge_ty& edges,
      size_t& totalWeight, size_t& totalIter) {

    kruskalAdjCopyBased<true, KruskalAdjNodeBased<true, KNodeAdj> > (nodes, edges, totalWeight, totalIter);
  }
};


class KruskalAdjNodeBasedUnsorted: public KruskalAdjBase<KNodeAdj> {
protected:

  virtual const std::string getVersion () const { return "Adjacency-based, Unsorted, Match Loop is Node-based"; }

  virtual void runMST (VecKNode_ty& nodes, VecKEdge_ty& edges,
      size_t& totalWeight, size_t& totalIter) {

    kruskalAdjCopyBased<false, KruskalAdjNodeBased<false, KNodeAdj> > (nodes, edges, totalWeight, totalIter);
  }
};


class BoruvkaAdjSorted: public KruskalAdjBase<KNodeAdj> {
protected:

  virtual const std::string getVersion () const { return "Boruvka Adjacency-based, Sorted"; }

  virtual void runMST (VecKNode_ty& nodes, VecKEdge_ty& edges,
      size_t& totalWeight, size_t& totalIter) {

    kruskalAdjCopyBased<true, BoruvkaAdj<true, KNodeAdj> > (nodes, edges, totalWeight, totalIter);
  }
};


class BoruvkaAdjUnsorted: public KruskalAdjBase<KNodeAdj> {
protected:

  virtual const std::string getVersion () const { return "Boruvka Adjacency-based, Unsorted"; }

  virtual void runMST (VecKNode_ty& nodes, VecKEdge_ty& edges,
      size_t& totalWeight, size_t& totalIter) {

    kruskalAdjCopyBased<false, BoruvkaAdj<false, KNodeAdj> > 
      (nodes, edges, totalWeight, totalIter);
  }
};
#endif //  _KRUSKAL_ADJ_H_

// template <bool sorted_tp, typename Flavor_tp>
// void kruskalAdj (
    // std::vector<KNode_tp*>& nodes,
    // VecKEdge_ty& edges,
    // size_t& totalWeight,
    // size_t& totalIter) {
// 
  // typedef typename Flavor_tp::MatchListTy MatchListTy;
  // typedef typename Flavor_tp::MatchLoopTy MatchLoopTy;
  // typedef typename Flavor_tp::MergeLoopTy MergeLoopTy;
// 
// 
// 
  // for (std::vector<KNode_tp*>::iterator i = nodes.begin (), ei = nodes.end ();
      // i != ei; ++i) {
// 
    // kruskal::arrangeAdj<sorted_tp> (**i);
// 
  // }
// 
// 
  // const bool SHOW_PARAMETER = true;
// 
  // std::ofstream* statsFile = NULL;
// 
  // if (SHOW_PARAMETER) {
    // statsFile = new std::ofstream ("parameter_kruskal.csv");
    // (*statsFile) << "LOOPNAME, STEP, PARALLELISM, WORKLIST_SIZE" << std::endl;
  // }
// 
// 
  // MatchListTy* matchList = new MatchListTy ();
  // MatchListTy* nextRoundList = new MatchListTy ();
  // MergeListTy mergeList;
// 
// 
  // Flavor_tp flavor (nodes, edges);
// 
  // flavor.fillInitial (nodes, edges, *matchList);
// 
  // totalIter = 0;
  // totalWeight = 0;
  // size_t numUnions = 0;
  // unsigned round = 0;
// 
  // Accumulator_ty matchIter (0);
  // Accumulator_ty mstSum (0);
  // Accumulator_ty mergeIter (0);
// 
  // galois::TimeAccumulator matchTimer;
  // galois::TimeAccumulator mergeTimer;
// 
  // while (true) {
// 
    // matchIter.reset (0);
    // matchTimer.start ();
// 
    // // galois::for_each_wl<MatchListTy> (*matchList, flavor.makeMatchLoop (*nextRoundList, mergeList, matchIter), "match_loop");
// 
    // matchTimer.stop ();
    // totalIter += matchIter.get ();
// 
// 
// 
    // mstSum.reset (0);
    // mergeIter.reset (0);
    // mergeTimer.start ();
// 
    // 
    // // galois::for_each_wl<MergeListTy> (mergeList, 
        // // flavor.makeMergeLoop (*nextRoundList, mstSum, mergeIter), 
        // // "merge_loop");
// 
    // mergeTimer.stop ();
    // totalWeight += mstSum.get ();
    // numUnions += mergeIter.get ();
// 
// 
    // assert (mergeIter.get () != 0 && "No merges? No progress?");
// 
    // if (SHOW_PARAMETER) {
      // (*statsFile) << "merge, " << round << ", " << mergeIter.get () 
        // << ", " << matchIter.get () << std::endl;
    // }
// 
    // // std::cout << "Match Attemps:    " << matchIter.get () << std::endl;
    // // std::cout << "Number of merges: " << mergeIter.get () << std::endl;
    // // std::cout << "Max size of adjacency list recorded: " << kruskal::getMaxAdjSize () << std::endl;
// 
    // ++round;
// 
// 
    // if (numUnions == (nodes.size () -1)) {
      // break;
    // }
// 
    // // switch workLists
    // delete matchList;
    // matchList = nextRoundList;
    // nextRoundList = new MatchListTy();
// 
    // flavor.finishRound ();
    // 
  // }
// 
  // std::cout << "Number of match-merge rounds: " << round << std::endl;
// 
  // std::cout << "Max size of adjacency list recorded: " << kruskal::getMaxAdjSize () << std::endl;
// 
  // std::cout << "Total time taken by Match Loop: " << matchTimer.get () << std::endl;;
  // std::cout << "Total time taken by Merge Loop: " << mergeTimer.get () << std::endl;;
// 
  // // after any k iterations matchList and nextRoundList point to 
  // // valid freeable locations
  // delete matchList;
  // delete nextRoundList;
// 
  // if (SHOW_PARAMETER) {
    // statsFile->close ();
    // delete statsFile; statsFile = NULL;
  // }
// 
// }

