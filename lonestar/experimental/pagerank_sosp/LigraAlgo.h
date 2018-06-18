#ifndef APPS_PAGERANK_LIGRAALGO_H
#define APPS_PAGERANK_LIGRAALGO_H

#include "galois/DomainSpecificExecutors.h"
#include "galois/graphs/OCGraph.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/GraphNodeBag.h"

#include <boost/mpl/if.hpp>

#include "PageRankOld.h"

template <bool UseGraphChi>
struct LigraAlgo : public galois::ligraGraphChi::ChooseExecutor<UseGraphChi> {
  typedef typename galois::graphs::LC_CSR_Graph<PNode, void>::
      template with_numa_alloc<true>::type ::template with_no_lockable<
          true>::type InnerGraph;
  typedef typename boost::mpl::if_c<
      UseGraphChi, galois::graphs::OCImmutableEdgeGraph<PNode, void>,
      galois::graphs::LC_InOut_Graph<InnerGraph>>::type Graph;
  typedef typename Graph::GraphNode GNode;

  std::string name() const { return UseGraphChi ? "LigraChi" : "Ligra"; }

  galois::GReduceMax<float> max_delta;
  galois::GAccumulator<size_t> small_delta;
  galois::GAccumulator<float> sum_delta;

  void readGraph(Graph& graph) {
    // Using dense forward option, so we don't need in-edge information
    galois::graphs::readGraph(graph, filename);
    this->checkIfInMemoryGraph(graph, memoryLimit);
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g) : g(g) {}
    void operator()(typename Graph::GraphNode n) const {
      PNode& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
      data.value  = 1.0;
      data.accum.write(0.0);
    }
  };

  struct EdgeOperator {
    template <typename GTy>
    bool cond(GTy& graph, typename GTy::GraphNode) {
      return true;
    }

    template <typename GTy>
    bool operator()(GTy& graph, typename GTy::GraphNode src,
                    typename GTy::GraphNode dst,
                    typename GTy::edge_data_reference) {
      PNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      int neighbors =
          std::distance(graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
                        graph.edge_end(src, galois::MethodFlag::UNPROTECTED));
      PNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      float delta  = sdata.value / neighbors;

      ddata.accum.atomicIncrement(delta);
      return false; // Topology-driven
    }
  };

  struct UpdateNode {
    LigraAlgo* self;
    Graph& graph;
    UpdateNode(LigraAlgo* s, Graph& g) : self(s), graph(g) {}
    void operator()(GNode src) const {
      PNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      float value  = (1.0 - alpha) * sdata.accum.read() + alpha;
      float diff   = std::fabs(value - sdata.value);
      if (diff <= tolerance)
        self->small_delta += 1;
      self->max_delta.update(diff);
      self->sum_delta.update(diff);
      sdata.value = value;
      sdata.accum.write(0);
    }
  };

  void operator()(Graph& graph) {
    galois::graphsNodeBagPair<> bags(graph.size());

    unsigned iteration = 0;

    // Initialize
    this->outEdgeMap(memoryLimit, graph, EdgeOperator(), bags.next());
    galois::do_all(graph, UpdateNode(this, graph));

    while (true) {
      iteration += 1;
      float delta   = max_delta.reduce();
      size_t sdelta = small_delta.reduce();
      std::cout << "iteration: " << iteration
                << " sum delta: " << sum_delta.reduce()
                << " max delta: " << delta << " small delta: " << sdelta << " ("
                << sdelta / (float)graph.size() << ")"
                << "\n";
      if (delta <= tolerance || iteration >= maxIterations)
        break;
      max_delta.reset();
      small_delta.reset();
      sum_delta.reset();
      // bags.swap();

      // this->outEdgeMap(memoryLimit, graph, EdgeOperator(), bags.cur(),
      // bags.next(), true);
      this->outEdgeMap(memoryLimit, graph, EdgeOperator(), bags.next());
      galois::do_all(graph, UpdateNode(this, graph));
    }

    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }
  }
};

#endif
