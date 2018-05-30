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

#ifndef BILLIARDS_SPEC_H
#define BILLIARDS_SPEC_H

#include "galois/PerThreadContainer.h"

#include "galois/graphs/Graph.h"

#include "galois/runtime/ROBexecutor.h"

#include "Billiards.h"
#include "BilliardsLevelExec.h"

class BilliardsSpec: public Billiards {
  using Graph = galois::graphs::FirstGraph<void*, void, true>;
  using GNode = Graph::GraphNode;
  using VecNodes = std::vector<GNode>;
  using AddListTy = galois::PerThreadVector<Event>;


  struct OpFunc {

    Table& table;
    const FP& endtime;
    Accumulator& iter;

    OpFunc (
        Table& table,
        const FP& endtime,
        Accumulator& iter)
      :
        table (table),
        endtime (endtime),
        iter (iter)
    {}


    template <typename C>
    void operator () (Event e, C& ctx) {
      // using const Event& in above param list gives error in the lambda (despite
      // using mutable), why???

      // std::cout << "Processing event: " << e.str () << std::endl;


      const bool notStale = e.notStale ();

      Ball* b1 = nullptr;
      Ball* b2 = nullptr;

      if (notStale) {
        auto alloc = ctx.getPerIterAlloc ();

        using BallAlloc = galois::PerIterAllocTy::rebind<Ball>::other;
        BallAlloc ballAlloc (alloc);

        b1 = ballAlloc.allocate (1);
        ballAlloc.construct (b1, *(e.getBall ()));


        if (e.getKind () == Event::BALL_COLLISION) {
          b2 = ballAlloc.allocate (1);
          ballAlloc.construct (b2, *(e.getOtherBall ()));

          Event copyEvent = Event::makeBallCollision (b1, b2, e.getTime ());
          copyEvent.simulate ();

        } else if (e.getKind () == Event::CUSHION_COLLISION) {
          Cushion* c = e.getCushion ();
          Event copyEvent = Event::makeCushionCollision (b1, c, e.getTime ());
          copyEvent.simulate ();

        }
      }

      auto oncommit = [this, &ctx, e, notStale, b1, b2] (void) mutable {


        if (notStale) {
          // update the state of the balls
          assert (b1 != nullptr);
          e.updateFirstBall (*b1);

          if (e.getKind () == Event::BALL_COLLISION) {
            assert (b2 != nullptr);
            e.updateOtherBall (*b2);
          }
        }

        using AddListTy = std::vector<Event, galois::PerIterAllocTy::rebind<Event>::other>;
        auto alloc = ctx.getPerIterAlloc ();
        AddListTy addList (alloc);
        table.addNextEvents (e, addList, endtime);

        for (auto i = addList.begin ()
            , endi = addList.end (); i != endi; ++i) {

          ctx.push (*i);
        }

        iter += 1;

      };

      ctx.addCommitAction (oncommit);
      
    }
  };

  void createLocks (const Table& table, Graph& graph, VecNodes& nodes) {
    nodes.reserve (table.getNumBalls ());

    for (unsigned i = 0; i < table.getNumBalls (); ++i) {
      nodes.push_back (graph.createNode (nullptr));
    }

  };

public:

  virtual const std::string version () const { return "using Speculative Executor"; }

  size_t runSim (Table& table, std::vector<Event>& initEvents, const FP& endtime, bool enablePrints=false) {

    Graph graph;
    VecNodes nodes;

    Accumulator iter;

    createLocks (table, graph, nodes);

    galois::runtime::for_each_ordered_rob (
        galois::runtime::makeStandardRange(initEvents.begin (), initEvents.end ()),
        Event::Comparator (),
        BilliardsLevelExec::VisitNhood (graph, nodes),
        OpFunc (table, endtime, iter));

    return iter.reduce ();

  }

};

#endif // BILLIARDS_SPEC_H
