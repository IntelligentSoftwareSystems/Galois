#ifndef GRAPH_COLORING_DETERMINISTIC_H
#define GRAPH_COLORING_DETERMINISTIC_H

#include "GraphColoringBase.h"

#include "Galois/Runtime/KDGparaMeter.h"
#include "Galois/Substrate/CompilerSpecific.h"

#include <atomic>
#include <random>

struct NodeDataDAG {
  unsigned color;
  std::atomic<unsigned> indegree;
  unsigned priority;
  unsigned id;

  NodeDataDAG (unsigned _id=0): color (0), indegree (0), priority (0), id (_id)
  {}

};

typedef Galois::Graph::LC_CSR_Graph<NodeDataDAG, void>
          ::with_numa_alloc<true>::type 
          // ::with_no_lockable<true>::type Graph;
          ::with_no_lockable<false>::type Graph;

typedef Graph::GraphNode GNode; 


class GraphColoringDAG: public GraphColoringBase<Graph> {
protected:

  struct NodeDataComparator {
    static bool compare (const NodeDataDAG& left, const NodeDataDAG& right) {
      if (left.priority != right.priority) {
        return left.priority < right.priority;
      } else {
        return left.id < right.id;
      }
    }

    bool operator () (const NodeDataDAG& left, const NodeDataDAG& right) const {
      return compare (left, right);
    }
  };


  template <typename W>
  void initDAG (W& initWork) {
    NodeDataComparator cmp;

    Galois::do_all_choice (
        Galois::Runtime::makeLocalRange (graph),
        [&] (GNode src) {
          auto& sd = graph.getData (src, Galois::MethodFlag::UNPROTECTED);

          // std::printf ("Processing node %d with priority %d\n", sd.id, sd.priority);

          unsigned addAmt = 0;
          for (Graph::edge_iterator e = graph.edge_begin (src, Galois::MethodFlag::UNPROTECTED),
              e_end = graph.edge_end (src, Galois::MethodFlag::UNPROTECTED); e != e_end; ++e) {
            GNode dst = graph.getEdgeDst (e);
            auto& dd = graph.getData (dst, Galois::MethodFlag::UNPROTECTED);

            if (cmp (dd, sd)) { // dd < sd
              ++addAmt;
            }
          }

          // only modify the node being processed
          // if we modify neighbors, each node will be 
          // processed twice.
          sd.indegree += addAmt;

          if (addAmt == 0) {
            assert (sd.indegree == 0);
            initWork.push (src);
          }
        },
        "init-dag",
        Galois::chunk_size<DEFAULT_CHUNK_SIZE> ());

  }

  struct ColorNodeDAG {
    typedef int tt_does_not_need_aborts;
    GraphColoringDAG& outer;

    template <typename C>
    void operator () (GNode src, C& ctx) {

      Graph& graph = outer.graph;

      auto& sd = graph.getData (src, Galois::MethodFlag::UNPROTECTED);
      assert (sd.indegree == 0);

      outer.colorNode (src);

      for (Graph::edge_iterator e = graph.edge_begin (src, Galois::MethodFlag::UNPROTECTED),
          e_end = graph.edge_end (src, Galois::MethodFlag::UNPROTECTED); e != e_end; ++e) {

        GNode dst = graph.getEdgeDst (e);
        auto& dd = graph.getData (dst, Galois::MethodFlag::UNPROTECTED);
        // std::printf ("Neighbor %d has indegree %d\n", dd.id, unsigned(dd.indegree));
        unsigned x = --(dd.indegree);
        if (x == 0) {
          ctx.push (dst);
        }
      }
    }
  };

  void colorDAG (void) {

    Galois::InsertBag<GNode> initWork;

    Galois::StatTimer t_dag_init("dag initialization time: ");

    t_dag_init.start ();
    initDAG (initWork);
    t_dag_init.stop ();

    typedef Galois::WorkList::AltChunkedFIFO<DEFAULT_CHUNK_SIZE> WL_ty;

    std::printf ("Number of initial sources: %zd\n", 
        std::distance (initWork.begin (), initWork.end ()));

    Galois::StatTimer t_dag_color ("dag coloring time: ");

    t_dag_color.start ();
    Galois::for_each_local (initWork, ColorNodeDAG {*this}, 
        Galois::loopname ("color-DAG"), Galois::wl<WL_ty> ());
    t_dag_color.stop ();
  }

  struct VisitNhood {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    GraphColoringDAG& outer;

    template <typename C>
    void operator () (GNode src, C&) {
      Graph& graph = outer.graph;
      NodeData& sd = graph.getData (src, Galois::MethodFlag::WRITE);
      for (Graph::edge_iterator e = graph.edge_begin (src, Galois::MethodFlag::WRITE),
          e_end = graph.edge_end (src, Galois::MethodFlag::WRITE); e != e_end; ++e) {
        GNode dst = graph.getEdgeDst (e);
      }
    }
  };

  struct ApplyOperator {
    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;
    GraphColoringDAG& outer;

    template <typename C>
    void operator () (GNode src, C&) {
      outer.colorNode (src);
    }
  };

  void colorKDGparam (void) {

    struct NodeComparator {
      Graph& graph;

      bool operator () (GNode ln, GNode rn) const {
        const auto& ldata = graph.getData (ln, Galois::MethodFlag::UNPROTECTED);
        const auto& rdata = graph.getData (rn, Galois::MethodFlag::UNPROTECTED);
        return NodeDataComparator::compare (ldata, rdata);
      }
    };


    Galois::Runtime::for_each_ordered_2p_param (
        Galois::Runtime::makeLocalRange (graph),
        NodeComparator {graph},
        VisitNhood {*this},
        ApplyOperator {*this},
        "coloring-ordered-param");
  }

  virtual void colorGraph (void) {
    assignPriority ();

    if (useParaMeter) {
      colorKDGparam ();
    } else {
      colorDAG ();
    }
  }

};


#endif // GRAPH_COLORING_DETERMINISTIC_H
