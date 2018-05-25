/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef KRUSKAL_FUNC_H_
#define KRUSKAL_FUNC_H_

#include <cassert>

#include "kruskalData.h"


namespace kruskal {

  template <typename KNode_tp>
  void checkRepValid (const KNode_tp* node) {
    assert (node != NULL && node->rep != NULL && "Node or its rep cannot be NULL");
  }

  
  template <typename KNode_tp>
  void checkIsRep (const KNode_tp* node) {
    checkRepValid (node);
    assert (node->rep == node && "Node is not representative");
  }

  //! find with path compression
  template <typename KNode_tp>
  KNode_tp* findPC (KNode_tp* node) {
    checkRepValid (node);
    if (node->rep == node) {
      return node;

    } else {
      node->rep = findPC (node->rep);
      checkIsRep (node->rep);
      return node->getRep ();
    }
  }


  //! read only find
  // introducing template parameter relaxed_tp
  // for the case when findRO is running in parallel
  // with union. In such cases, findRO may return the 
  // result just before union on the node takes place
  // and changes its rep, hence throwing an asserion error
  // in checkIsRep. 
  template <bool relaxed_tp, typename KNode_tp>
  const KNode_tp* _findRO (const KNode_tp* node) {
    checkRepValid (node);

    KNode_tp* n = const_cast<KNode_tp*> (node);

    while (n != n->rep) {
      n = n->getRep ();
    }

    if (!relaxed_tp) {
      checkIsRep (n);
    }

    return n;
  }


  template <typename KNode_tp>
  const KNode_tp* findRO (const KNode_tp* node) {
    return _findRO<false> (node);
  }

  template <typename KNode_tp>
  const KNode_tp* findROrelax (const KNode_tp* node) {
    return _findRO<true> (node);
  }

  //! link up other to master, without modifying master
  template <typename KNode_tp>
  void linkUp (KNode_tp* other, KNode_tp* master) {
    checkIsRep (other);
    // checkIsRep (master);, can't check this in parallel

    assert (master != other);

    other->rep = master;
  }


  //! @return pair<rep, other> where rep is representative component of the unioned components
  //     and other is the losing component
  template <typename KNode_tp>
  std::pair<KNode_tp*, KNode_tp*>  unionByRank (KNode_tp* comp1, KNode_tp* comp2) {
    checkIsRep (comp1);
    checkIsRep (comp2);

    assert (comp1 != comp2);

    if (comp1->rank > comp2->rank ) {

      comp2->rep = comp1;
      return std::make_pair (comp1, comp2);

    } else {

      comp1->rep = comp2;
      
      if (comp1->rank == comp2->rank) {
        ++comp2->rank;
      }

      return std::make_pair (comp2, comp1);
    }

  }

  template <typename KNode_tp>
  bool contract (KEdge<KNode_tp>& e) {
    KNode_tp* rep1 = kruskal::findPC (e.src);
    KNode_tp* rep2 = kruskal::findPC (e.dst);

    if (rep1 != rep2) {
      kruskal::unionByRank (rep1, rep2);
      e.inMST = true;
      return true;

    } else {
      assert (e.inMST == false);
      return false;
    }
  }



  template <bool relaxed_tp, typename KNode_tp>
  struct _NotSelfEdge {

    bool operator () (const KEdge<KNode_tp>* e) const {
      return (_findRO<relaxed_tp> (e->src) != _findRO<relaxed_tp> (e->dst));
    }
  };

  template <typename KNode_tp> 
  struct NotSelfEdge: public _NotSelfEdge<false, KNode_tp> {};

  template <typename KNode_tp>
  struct NotSelfEdgeRelax: public _NotSelfEdge<true, KNode_tp> {};


  // has a relaxed version due to NotSelfEdge
  template <bool relaxed_tp, typename KNode_tp>
  void minToFront (KNode_tp& node) {

    typename KNode_tp::EdgeListTy::iterator min = 
      std::min_element (node.edges.begin (), node.edges.end (), typename KEdge<KNode_tp>::PtrComparator ());

    assert (min != node.edges.end ());

    _NotSelfEdge<relaxed_tp, KNode_tp> f;
    assert (f (*min));

    node.edges.splice (node.edges.begin (), node.edges, min);
  }

  // for Unsorted case, we need to move
  // the lightest edge to the front, which
  // is what the getLightest looks for during match
  template <bool sorted_tp, typename KNode_tp>
  void arrangeAdj (KNode_tp& node) {
    if (sorted_tp) {
      node.edges.sort (typename KEdge<KNode_tp>::PtrComparator ());

    } else {
      minToFront<false> (node); // non-relaxed version
    }
  }


  // same implementation for both Sorted and Unsorted cases
  // The lightest outgoing edge is at the front of the list
  // and this invariant is ensured by merge

  template <bool sorted_tp, typename KNode_tp>
  KEdge<KNode_tp>* getLightest (KNode_tp& n) {
    assert (!n.edges.empty () && "calling front () on empty edges list???");

    KEdge<KNode_tp>* ret = n.edges.front ();
    assert (NotSelfEdge<KNode_tp> () (ret));
    return ret;
  }


  static size_t maxAdjSize = 0;

  size_t getMaxAdjSize () {
    return maxAdjSize;
  }

  template <bool sorted_tp, typename KNode_tp>
  void merge(KNode_tp& master, KNode_tp& other) {
    // assume components have been unioned
    // master is the representative of unioned component
    assert (other.rep == &master);
    assert (master.rep == &master);

    assert (!master.edges.empty ());
    assert (!other.edges.empty ());

    // for measurement only
    // if (maxAdjSize < (master.edges.size () + other.edges.size ())) {
      // maxAdjSize = (master.edges.size () + other.edges.size ());
    // }


    // first we merge the adj lists and then
    // we remove the self edges. Ideally we would like to remove
    // the self edges and free their memory, but for parallel performance
    // we want to avoid calls to allocator. Therefore, we partition
    // the adj list into self and non-self edges and move the non-self edges
    // to non-master node
    if (sorted_tp) {
      // merge performs a sorted merge
      master.edges.merge (other.edges, typename KEdge<KNode_tp>::PtrComparator ());
    } else {
      // simply abutt
      master.edges.splice (master.edges.end (), other.edges);
    }
    assert (other.edges.empty ());

    typename KNode_tp::EdgeListTy::iterator self_begin;

    if (sorted_tp) {
      // Unsorted version can use partition simply, but Sorted
      // version must use stable_partition to ensure the sorted order
      // does not get changed
      self_begin = std::stable_partition (master.edges.begin (), master.edges.end (), NotSelfEdgeRelax<KNode_tp> ());
      // relaxed version calls findROrelax because
      // results of findRO may be stale due to union
      // occuring in parallel.
    } else {
      self_begin = std::partition (master.edges.begin (), master.edges.end (), NotSelfEdgeRelax<KNode_tp> ());
    }

    // now move the rest of the range to other
    other.edges.splice (other.edges.begin (), master.edges, self_begin, master.edges.end ());



    if (!sorted_tp) {
      // finally, for unsorted case, we rearrange the adj list so that
      // minimum non self edge is moved to front
      // don't use size () on list
      if (!master.edges.empty ()) {
        // need to move lightest non-self edge to front of the list
        minToFront<true> (master); // running the relaxed version
      }
    }

  }



  // for anlaysis only

  template <typename KNode_tp>
  size_t findROcostModel (KNode_tp* node) {
    checkRepValid (node);

    size_t cost = 0;

    KNode_tp* n = node;

    while (n != n->rep) {
      n = n->rep;
      ++cost;
    }

    return cost;
  }

  template <typename KNode_tp>
  void _internalMergeCost (KNode_tp* rep, size_t& mergeCost, size_t& findROcost) {
    checkIsRep (rep);
    for (typename KNode_tp::EdgeListTy::iterator i = rep->edges.begin (), ei = rep->edges.end (); 
        i != ei; ++i) {
      ++mergeCost;

      findROcost += findROcostModel (i->src);
      findROcost += findROcostModel (i->dst);
    }
  }


  // mergeCost measures the size of the adjacency lists of the representatives
  // of the edges
  // findROcost is the sum of cost of each findRO on 
  template <typename KNode_tp>
  void measureMergeCost (KEdge<KNode_tp>& edge, size_t& mergeCost, size_t& findROcost) {
    KNode_tp* rep1 = edge.src->rep;
    KNode_tp* rep2 = edge.dst->rep;

    checkIsRep (rep1);
    checkIsRep (rep2);

    _internalMergeCost (rep1, mergeCost, findROcost);
    _internalMergeCost (rep2, mergeCost, findROcost);

  }




}
#endif //  KRUSKAL_FUNC_H_
