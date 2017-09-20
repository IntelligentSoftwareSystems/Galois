#ifndef BILLIARDS_PARALLEL_H
#define BILLIARDS_PARALLEL_H

#include "Billiards.h"
#include "dependTest.h"

#include "Galois/PerThreadContainer.h"
#include "Galois/Graphs/Graph.h"


using AddListTy = galois::PerThreadVector<Event>;

struct VisitNhoodSafetyTest {
  static const unsigned CHUNK_SIZE = 1;

  OrderDepTest dt;

  template <typename C, typename I>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event& event, const C& c, const I beg, const I end) {

    bool indep = true;

    for (I i = beg; i != end; ++i) {
      if (event > *i) {
        if (dt.dependsOn (event, *i)) {
          indep = false;
          break;
        }
      }
    }

    if (!indep) {
      galois::Runtime::signalConflict ();
    }
  }
};


template <typename Tbl_t, typename Graph, typename VecNodes>
void createLocks (const Tbl_t& table, Graph& graph, VecNodes& nodes) {
  nodes.reserve (table.getNumBalls ());

  for (unsigned i = 0; i < table.getNumBalls (); ++i) {
    nodes.push_back (graph.createNode (nullptr));
  }

};


template <typename Graph, typename VecNodes>
struct VisitNhoodLocks {

  static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

  Graph& graph;
  VecNodes& nodes;

  VisitNhoodLocks (Graph& graph, VecNodes& nodes): graph (graph), nodes (nodes) {}

  template <typename C>
  void operator () (const Event& e, C& ctx) const {

    Ball* b1 = e.getBall ();
    assert (b1->getID () < nodes.size ());
    graph.getData (nodes[b1->getID ()], galois::MethodFlag::WRITE);

    if (e.getKind () == Event::BALL_COLLISION) {
      Ball* b2 = e.getOtherBall ();
      assert (b2->getID () < nodes.size ());
      graph.getData (nodes[b2->getID ()], galois::MethodFlag::WRITE);
    }

  }
};


struct ExecSources {
  static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

  template <typename C>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event& e, C& ctx) const {
    const_cast<Event&> (e).simulate ();
  }
};

template <typename Tbl_t>
struct AddEvents {

  static const unsigned CHUNK_SIZE = 1;

  Tbl_t& table;
  const FP& endtime;
  AddListTy& addList;
  Accumulator& iter;
  bool enablePrints;

  AddEvents (
      Tbl_t& table,
      const FP& endtime,
      AddListTy& addList,
      Accumulator& iter,
      bool enablePrints)
    :
      table (table),
      endtime (endtime),
      addList (addList),
      iter (iter),
      enablePrints (enablePrints)
  {}


  template <typename C>
  GALOIS_ATTRIBUTE_PROF_NOINLINE void operator () (const Event& e, C& ctx) {

    addList.get ().clear ();

    // TODO: use locks to update balls' state atomically 
    // and read atomically
    // const_cast<Event&>(e).simulate ();
    table.addNextEvents (e, addList.get (), endtime);

    for (auto i = addList.get ().begin ()
        , endi = addList.get ().end (); i != endi; ++i) {

      ctx.push (*i);

      if (enablePrints) {
        std::cout << "Adding event=" << i->str () << std::endl;
      }
    }

    iter += 1;
  }
};


#endif // BILLIARDS_PARALLEL_H
