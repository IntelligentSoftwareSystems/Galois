/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

#ifndef KRUSKAL_ORDERED_H_
#define KRUSKAL_ORDERED_H_

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Callbacks.h"

#include "kruskalData.h"
#include "kruskalFunc.h"
#include "kruskal.h"

typedef galois::GAccumulator<size_t> Accumulator_ty;

template <typename KNode_tp>
struct EdgeComparator : public galois::CompareCallback {

  virtual bool compare(void* a, void* b) const {
    KEdge<KNode_tp>** ea = static_cast<KEdge<KNode_tp>**>(a);
    KEdge<KNode_tp>** eb = static_cast<KEdge<KNode_tp>**>(b);

    return (KEdge<KNode_tp>::PtrComparator::compare(*ea, *eb) < 0);
  }

  virtual bool operator()(void* a, void* b) const { return compare(a, b); }
};

struct KNodeLockable : public galois::runtime::Lockable, KNode {

  KNodeLockable(unsigned id) : galois::runtime::Lockable(), KNode(id) {}

  void acquire(galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
    galois::runtime::acquire(this, mflag);
  }

  KNodeLockable* getRep() { return static_cast<KNodeLockable*>(KNode::rep); }
};

class KruskalOrderedSrc : public Kruskal<KNodeLockable> {
  typedef Kruskal<KNodeLockable> Super_ty;

  struct MatchOperator {
    Accumulator_ty& matchIter;

    MatchOperator(Accumulator_ty& matchIter) : matchIter(matchIter) {}

    template <typename C>
    void operator()(KEdge<KNodeLockable>* pedge, C&) {
      matchIter += 1;

      assert(pedge != NULL);
      KEdge<KNodeLockable>& edge = *pedge;

      KNodeLockable* rep1 = kruskal::findPC(edge.src);
      KNodeLockable* rep2 = kruskal::findPC(edge.dst);

      rep1->acquire();
      rep2->acquire();
    }
  };

  struct UnionOperator {
    Accumulator_ty& mstSum;
    Accumulator_ty& mergeIter;

    UnionOperator(Accumulator_ty& _mstSum, Accumulator_ty& _mergeIter)
        : mstSum(_mstSum), mergeIter(_mergeIter) {}

    // void signalFailSafe () {
    // galois::runtime::checkWrite (galois::WRITE);
    // }

    template <typename C>
    void operator()(KEdge<KNodeLockable>* pedge, C&) {
      assert(pedge != NULL);
      KEdge<KNodeLockable>& edge = *pedge;

      KNodeLockable* rep1 = edge.src->getRep();
      KNodeLockable* rep2 = edge.dst->getRep();

      if (rep1 != rep2) {
        // std::cout << "Contracting: " << edge.str () << std::endl;
        kruskal::unionByRank(rep1, rep2);

        edge.inMST = true;
        mstSum += edge.weight;
        mergeIter += 1;
      }
    }
  };

protected:
  virtual const std::string getVersion() const {
    return "Kruskal using ordered runtime";
  }

  virtual void runMST(Super_ty::VecKNode_ty& nodes,
                      Super_ty::VecKEdge_ty& edges, size_t& totalWeight,
                      size_t& totalIter) {

    totalIter   = 0;
    totalWeight = 0;

    Accumulator_ty mstSum;
    Accumulator_ty matchIter;
    Accumulator_ty mergeIter;

    galois::StatTimer t_feach("for_each loop time: ");

    t_feach.start();
    galois::for_each_ordered(
        edges.begin(), edges.end(), MatchOperator(matchIter),
        UnionOperator(mstSum, mergeIter), EdgeComparator<KNodeLockable>());
    t_feach.stop();

    totalWeight = mstSum.reduce();
    totalIter   = matchIter.reduce();

    std::cout << "Match iterations: " << matchIter.reduce() << std::endl;
    std::cout << "Merge iterations: " << mergeIter.reduce() << std::endl;
  }
};

class KruskalOrderedNonSrc : public Kruskal<KNodeMin> {

  typedef Kruskal<KNodeMin> Super_ty;

  struct MatchOperator {
    Accumulator_ty& matchIter;

    MatchOperator(Accumulator_ty& _matchIter) : matchIter(_matchIter) {}

    template <typename C>
    void operator()(KEdge<KNodeMin>* edge, C&) {
      assert(edge != NULL);

      matchIter += 1;

      KNodeMin* rep1 = kruskal::findPC(edge->src);

      KNodeMin* rep2 = kruskal::findPC(edge->dst);

      if (rep1 != rep2) {

        rep1->claimAsMin(edge);
        rep2->claimAsMin(edge);
      }
    }
  };

  struct LinkUpOperator {
    Accumulator_ty& mstSum;
    Accumulator_ty& numUnions;
    Accumulator_ty& mergeIter;

    LinkUpOperator(Accumulator_ty& _mstSum, Accumulator_ty& _numUnions,
                   Accumulator_ty& _mergeIter)
        : mstSum(_mstSum), numUnions(_numUnions), mergeIter(_mergeIter) {}

    template <typename C>
    void operator()(KEdge<KNodeMin>* edge, C&) {
      assert(edge != NULL);
      // relies on find with path-compression
      KNodeMin* rep1 = edge->src->getRep();
      KNodeMin* rep2 = edge->dst->getRep();

      // not  a self-edge
      if (rep1 != rep2) {
        mergeIter += 1;

        bool succ1 = (rep1->minEdge == edge);
        bool succ2 = (rep2->minEdge == edge);

        if (succ1) {
          kruskal::linkUp(rep1, rep2);

        } else if (succ2) {
          kruskal::linkUp(rep2, rep1);

        } else {
          // defer processing of this edge to the next round
          galois::runtime::signalConflict();
        }

        if (succ1 || succ2) {
          // std::cout << "Contracting: " << edge->str () << std::endl;
          numUnions += 1;
          mstSum += edge->weight;
          edge->inMST = true;

          // reset minEdge for next round
          // only on success
          if (succ1) {
            rep1->minEdge = NULL;
          }

          if (succ2) {
            rep2->minEdge = NULL;
          }
        }
      }
    }
  };

protected:
  virtual const std::string getVersion() const {
    return "Kruskal non-src, using ordered runtime";
  }

  virtual void runMST(Super_ty::VecKNode_ty& nodes,
                      Super_ty::VecKEdge_ty& edges, size_t& totalWeight,
                      size_t& totalIter) {

    totalIter   = 0;
    totalWeight = 0;

    Accumulator_ty mstSum;
    Accumulator_ty numUnions;
    Accumulator_ty matchIter;
    Accumulator_ty mergeIter;

    galois::StatTimer t_feach("for_each loop time: ");

    t_feach.start();
    galois::for_each_ordered(edges.begin(), edges.end(),
                             MatchOperator(matchIter),
                             LinkUpOperator(mstSum, numUnions, mergeIter),
                             EdgeComparator<KNodeMin>());
    t_feach.stop();

    totalWeight = mstSum.reduce();
    totalIter   = matchIter.reduce();

    assert((numUnions.reduce() == (nodes.size() - 1)) &&
           "Wrong number of unions reported?");

    std::cout << "Match iterations: " << matchIter.reduce() << std::endl;
    std::cout << "Merge iterations: " << mergeIter.reduce() << std::endl;
    std::cout << "numUnions: " << numUnions.reduce() << std::endl;
  }
};

#endif //  KRUSKAL_RUNTIME_H_
