/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef GRAPH_COLORING_DETERMINISTIC_H
#define GRAPH_COLORING_DETERMINISTIC_H

#include "GraphColoringBase.h"

#include "galois/runtime/KDGparaMeter.h"
#include "galois/substrate/CompilerSpecific.h"

#include <atomic>
#include <random>

struct NodeDataDAG {
  unsigned color;
  std::atomic<unsigned> indegree;
  unsigned priority;
  unsigned id;

  NodeDataDAG(unsigned _id = 0) : color(0), indegree(0), priority(0), id(_id) {}
};

typedef galois::graphs::LC_CSR_Graph<NodeDataDAG,
                                     void>::with_numa_alloc<true>::type
    // ::with_no_lockable<true>::type Graph;
    ::with_no_lockable<false>::type Graph;

typedef Graph::GraphNode GNode;

class GraphColoringDAG : public GraphColoringBase<Graph> {
protected:
  struct NodeDataComparator {
    static bool compare(const NodeDataDAG& left, const NodeDataDAG& right) {
      if (left.priority != right.priority) {
        return left.priority < right.priority;
      } else {
        return left.id < right.id;
      }
    }

    bool operator()(const NodeDataDAG& left, const NodeDataDAG& right) const {
      return compare(left, right);
    }
  };

  template <typename W>
  void initDAG(W& initWork) {
    NodeDataComparator cmp;

    galois::do_all_choice(
        galois::runtime::makeLocalRange(graph),
        [&](GNode src) {
          auto& sd = graph.getData(src, galois::MethodFlag::UNPROTECTED);

          // std::printf ("Processing node %d with priority %d\n", sd.id,
          // sd.priority);

          unsigned addAmt = 0;
          for (Graph::edge_iterator
                   e = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
                   e_end = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
               e != e_end; ++e) {
            GNode dst = graph.getEdgeDst(e);
            auto& dd  = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

            if (cmp(dd, sd)) { // dd < sd
              ++addAmt;
            }
          }

          // only modify the node being processed
          // if we modify neighbors, each node will be
          // processed twice.
          sd.indegree += addAmt;

          if (addAmt == 0) {
            assert(sd.indegree == 0);
            initWork.push(src);
          }
        },
        "init-dag", galois::chunk_size<DEFAULT_CHUNK_SIZE>());
  }

  struct ColorNodeDAG {
    typedef int tt_does_not_need_aborts;
    GraphColoringDAG& outer;

    template <typename C>
    void operator()(GNode src, C& ctx) {

      Graph& graph = outer.graph;

      auto& sd = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      assert(sd.indegree == 0);

      outer.colorNode(src);

      for (Graph::edge_iterator
               e     = graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
               e_end = graph.edge_end(src, galois::MethodFlag::UNPROTECTED);
           e != e_end; ++e) {

        GNode dst = graph.getEdgeDst(e);
        auto& dd  = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
        // std::printf ("Neighbor %d has indegree %d\n", dd.id,
        // unsigned(dd.indegree));
        unsigned x = --(dd.indegree);
        if (x == 0) {
          ctx.push(dst);
        }
      }
    }
  };

  void colorDAG(void) {

    galois::InsertBag<GNode> initWork;

    galois::StatTimer t_dag_init("dag initialization time: ");

    t_dag_init.start();
    initDAG(initWork);
    t_dag_init.stop();

    typedef galois::worklists::PerThreadChunkFIFO<DEFAULT_CHUNK_SIZE> WL_ty;

    std::printf("Number of initial sources: %zd\n",
                std::distance(initWork.begin(), initWork.end()));

    galois::StatTimer t_dag_color("dag coloring time: ");

    t_dag_color.start();
    galois::for_each(initWork, ColorNodeDAG{*this},
                     galois::loopname("color-DAG"), galois::wl<WL_ty>());
    t_dag_color.stop();
  }

  struct VisitNhood {

    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;

    GraphColoringDAG& outer;

    template <typename C>
    void operator()(GNode src, C&) {
      Graph& graph = outer.graph;
      NodeData& sd = graph.getData(src, galois::MethodFlag::WRITE);
      for (Graph::edge_iterator
               e     = graph.edge_begin(src, galois::MethodFlag::WRITE),
               e_end = graph.edge_end(src, galois::MethodFlag::WRITE);
           e != e_end; ++e) {
        GNode dst = graph.getEdgeDst(e);
      }
    }
  };

  struct ApplyOperator {
    static const unsigned CHUNK_SIZE = DEFAULT_CHUNK_SIZE;
    GraphColoringDAG& outer;

    template <typename C>
    void operator()(GNode src, C&) {
      outer.colorNode(src);
    }
  };

  void colorKDGparam(void) {

    struct NodeComparator {
      Graph& graph;

      bool operator()(GNode ln, GNode rn) const {
        const auto& ldata = graph.getData(ln, galois::MethodFlag::UNPROTECTED);
        const auto& rdata = graph.getData(rn, galois::MethodFlag::UNPROTECTED);
        return NodeDataComparator::compare(ldata, rdata);
      }
    };

    galois::runtime::for_each_ordered_2p_param(
        galois::runtime::makeLocalRange(graph), NodeComparator{graph},
        VisitNhood{*this}, ApplyOperator{*this}, "coloring-ordered-param");
  }

  virtual void colorGraph(void) {
    assignPriority();

    if (useParaMeter) {
      colorKDGparam();
    } else {
      colorDAG();
    }
  }
};

#endif // GRAPH_COLORING_DETERMINISTIC_H
