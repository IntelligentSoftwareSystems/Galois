#ifndef GRAPH_COLORING_DETERMINISTIC_H
#define GRAPH_COLORING_DETERMINISTIC_H

#include "GraphColoringBase.h"

#include "Galois/Runtime/KDGparaMeter.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include <atomic>

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
    bool operator () (const NodeDataDAG& left, const NodeDataDAG& right) const {
      if (left.priority != right.priority) {
        return left.priority < right.priority;
      } else {
        return left.id < right.id;
      }
    }
  };

  template <typename F>
  void assignPriorityHelper (const F& nodeFunc) {
    Galois::do_all_choice (
        Galois::Runtime::makeLocalRange (graph),
        [&] (GNode node) {
          nodeFunc (node);
        },
        "assign-priority",
        Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());
  }

  void assignPriority (void) {

    auto byId = [&] (GNode node) {
      auto& nd = graph.getData (node, Galois::NONE);
      nd.priority = nd.id;
    };

    auto minDegree = [&] (GNode node) {
      auto& nd = graph.getData (node, Galois::NONE);
      nd.priority = std::distance (
                      graph.edge_begin (node, Galois::NONE),
                      graph.edge_end (node, Galois::NONE));
    };

    const size_t numNodes = graph.size ();
    auto maxDegree = [&] (GNode node) {
      auto& nd = graph.getData (node, Galois::NONE);
      nd.priority = numNodes - std::distance (
                                  graph.edge_begin (node, Galois::NONE),
                                  graph.edge_end (node, Galois::NONE));
    };
    

    switch (heuristic) {
      case FIRST_FIT:
        assignPriorityHelper (byId);
        break;

      case MIN_DEGREE:
        assignPriorityHelper (minDegree);
        break;

      case MAX_DEGREE:
        assignPriorityHelper (maxDegree);
        break;

      default:
        std::abort ();
    }
  }

  void initDAG (void) {
    NodeDataComparator cmp;

    Galois::do_all_choice (
        Galois::Runtime::makeLocalRange (graph),
        [&] (GNode src) {
          auto& sd = graph.getData (src, Galois::NONE);

          // std::printf ("Processing node %d with priority %d\n", sd.id, sd.priority);

          for (Graph::edge_iterator e = graph.edge_begin (src, Galois::NONE),
              e_end = graph.edge_end (src, Galois::NONE); e != e_end; ++e) {
            GNode dst = graph.getEdgeDst (e);
            auto& dd = graph.getData (dst, Galois::NONE);

            if (cmp (dd, sd)) { // dd < sd
              ++(sd.indegree);
            } // only modify the node being processed
            // if we modify neighbors, each node will be 
            // processed twice.
          }
        },
        "init-dag",
        Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());

  }

  struct ColorNodeDAG {
    typedef int tt_does_not_need_aborts;
    GraphColoringDAG& outer;

    template <typename C>
    void operator () (GNode src, C& ctx) {

      Graph& graph = outer.graph;

      auto& sd = graph.getData (src, Galois::NONE);
      assert (sd.indegree == 0);

      outer.colorNode (src);

      for (Graph::edge_iterator e = graph.edge_begin (src, Galois::NONE),
          e_end = graph.edge_end (src, Galois::NONE); e != e_end; ++e) {

        GNode dst = graph.getEdgeDst (e);
        auto& dd = graph.getData (dst, Galois::NONE);
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

    Galois::do_all_choice (
        Galois::Runtime::makeLocalRange (graph),
        [&] (GNode src) {
          auto& sd = graph.getData (src, Galois::NONE);
          if (sd.indegree == 0) {
            initWork.push (src);
          }
        },
        "init-worklist",
        Galois::doall_chunk_size<DEFAULT_CHUNK_SIZE> ());

    typedef Galois::WorkList::AltChunkedFIFO<DEFAULT_CHUNK_SIZE> WL_ty;

    std::printf ("Number of initial sources: %zd\n", 
        std::distance (initWork.begin (), initWork.end ()));

    Galois::for_each_local (initWork, ColorNodeDAG {*this}, 
        Galois::loopname ("color-DAG"), Galois::wl<WL_ty> ());
  }

  struct VisitNhood {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    GraphColoringDAG& outer;

    template <typename C>
    void operator () (GNode src, C&) {
      Graph& graph = outer.graph;
      NodeData& sd = graph.getData (src, Galois::CHECK_CONFLICT);
      for (Graph::edge_iterator e = graph.edge_begin (src, Galois::CHECK_CONFLICT),
          e_end = graph.edge_end (src, Galois::CHECK_CONFLICT); e != e_end; ++e) {
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
      const NodeDataComparator dataCmp;

      bool operator () (GNode ln, GNode rn) const {
        const auto& ldata = graph.getData (ln, Galois::NONE);
        const auto& rdata = graph.getData (rn, Galois::NONE);
        return dataCmp (ldata, rdata);
      }
    };


    Galois::Runtime::for_each_ordered_2p_param (
        Galois::Runtime::makeLocalRange (graph),
        NodeComparator {graph, NodeDataComparator {}},
        VisitNhood {*this},
        ApplyOperator {*this},
        "coloring-ordered-param");
  }

  virtual void colorGraph (void) {
    assignPriority ();
    const bool paraMeter = true;

    if (paraMeter) {
      colorKDGparam ();
    } else {
      initDAG ();
      colorDAG ();
    }
  }

};


#endif // GRAPH_COLORING_DETERMINISTIC_H
