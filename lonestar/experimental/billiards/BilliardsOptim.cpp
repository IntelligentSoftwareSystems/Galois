/** Billiards Simulation using speculative executor-*- C++ -*-
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
 * @author <ahassaan@ices.utexas.edu>
 */


#include "Galois/PerThreadContainer.h"

#include "Galois/Graphs/Graph.h"

#include "Galois/Runtime/ROBexecutor.h"
#include "Galois/Runtime/OrderedSpeculation.h"

#include "Billiards.h"
#include "BilliardsParallel.h"

class BilliardsOptim: public Billiards<BilliardsOptim, Table<BallOptim<> > > {

  using Graph = galois::Graph::FirstGraph<void*, void, true>;
  using GNode = Graph::GraphNode;
  using VecNodes = std::vector<GNode>;
  using AddListTy = galois::PerThreadVector<Event>;

  using Tbl_t = Table<BallOptim<> >;
  using Ball_t = typename Tbl_t::Ball_t;

  struct ExecSourcesOptim {
    static const unsigned CHUNK_SIZE = 4;

    Tbl_t& table;
    bool enablePrints;
    bool logEvents;

    template <typename C>
    GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event& e, C& ctx) const {


      if (enablePrints) {
        std::cout << "Processing event=" << e.str () << std::endl;
      }

      using Wrapper = typename Ball_t::Wrapper;

      Wrapper* bw1 = static_cast<Ball_t*> (e.getBall ())->getWrapper ();
      Wrapper* bw2 = nullptr;

      if (e.getKind () == Event::BALL_COLLISION) {
        bw2 = static_cast<Ball_t*> (e.getOtherBall ())->getWrapper ();
      }

      Ball_t* b1 = bw1->checkpoint (e);
      Ball_t* b2 = nullptr;
      
      if (bw2) {
        b2 = bw2->checkpoint (e);
      }

      auto ccopy = e.getCounterCopies ();

      auto restore = [bw1, bw2, b1, b2, &e, ccopy] (void) {
        bw1->restore (b1);
        
        if (bw2) {
          assert (b2);
          bw2->restore (b2);
        }

        const_cast<Event&> (e).restoreCounters (ccopy);
      };

      ctx.addUndoAction (restore);

      auto reclaim = [bw1, bw2, b1, b2, &e, this] (void) {
        bw1->reclaim (e, b1);

        if (bw2) {
          assert (b2);
          bw2->reclaim (e, b2);
        }

        if (logEvents) {
          table.logCollisionEvent (e);
        }

        if (enablePrints) {
          std::cout << "Committing event=" << e.str () << std::endl;
        }
      };

      ctx.addCommitAction (reclaim);

      if (e.notStale ()) {
        // assert (Collision::isValidCollision (e));
      }

      if (e.notStale () && Collision::isValidCollision (e)) {
        const_cast<Event&> (e).simulate ();
      }
    }
  };



public:

  virtual const std::string version () const { return "using Speculative Executor"; }

  size_t runSim (Tbl_t& table, std::vector<Event>& initEvents, const FP& endtime, const bool enablePrints=false, const bool logEvents=false) {

    Graph graph;
    VecNodes nodes;

    Accumulator iter;
    AddListTy addList;

    createLocks (table, graph, nodes);

    galois::runtime::for_each_ordered_spec (
        galois::runtime::makeStandardRange(initEvents.begin (), initEvents.end ()),
        Event::Comparator (),
        VisitNhoodLocks<Graph, VecNodes> (graph, nodes),
        ExecSourcesOptim {table, enablePrints, logEvents},
        AddEvents<Tbl_t> (table, endtime, addList, iter, enablePrints),
        std::make_tuple (galois::loopname ("billiards-optimistic")));


    for (unsigned i = 0; i < table.getNumBalls (); ++i) {
      
      auto bw = table.getBallByID (i).getWrapper ();
      assert (bw->hasEmptyHistory ());
    }

    if (enablePrints) {
      table.printState (std::cout);
    }

    return iter.reduce ();

  }

};

int main (int argc, char* argv[]) {
  BilliardsOptim s;
  s.run (argc, argv);
  return 0;
}
