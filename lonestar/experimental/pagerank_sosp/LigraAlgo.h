#ifndef APPS_PAGERANK_LIGRAALGO_H
#define APPS_PAGERANK_LIGRAALGO_H

#include "Galois/DomainSpecificExecutors.h"
#include "Galois/Graphs/OCGraph.h"
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/GraphNodeBag.h"

#include <boost/mpl/if.hpp>

#include "PageRankOld.h"

template<bool UseGraphChi>
struct LigraAlgo: public Galois::LigraGraphChi::ChooseExecutor<UseGraphChi> {
  typedef typename Galois::Graph::LC_CSR_Graph<PNode,void>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type
    InnerGraph;
  typedef typename boost::mpl::if_c<UseGraphChi,
          Galois::Graph::OCImmutableEdgeGraph<PNode,void>,
          Galois::Graph::LC_InOut_Graph<InnerGraph>>::type
          Graph;
  typedef typename Graph::GraphNode GNode;

  std::string name() const { return UseGraphChi ? "LigraChi" : "Ligra"; }
  
  Galois::GReduceMax<float> max_delta;
  Galois::GAccumulator<size_t> small_delta;
  Galois::GAccumulator<float> sum_delta;

  void readGraph(Graph& graph) {
    // Using dense forward option, so we don't need in-edge information
    Galois::Graph::readGraph(graph, filename); 
    this->checkIfInMemoryGraph(graph, memoryLimit);
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(typename Graph::GraphNode n) const {
      PNode& data = g.getData(n, Galois::MethodFlag::UNPROTECTED);
      data.value = 1.0;
      data.accum.write(0.0);
    }
  };

  struct EdgeOperator {
    template<typename GTy>
    bool cond(GTy& graph, typename GTy::GraphNode) { return true; }

    template<typename GTy>
    bool operator()(GTy& graph, typename GTy::GraphNode src, typename GTy::GraphNode dst, typename GTy::edge_data_reference) {
      PNode& sdata = graph.getData(src, Galois::MethodFlag::UNPROTECTED);
      int neighbors = std::distance(graph.edge_begin(src, Galois::MethodFlag::UNPROTECTED),
          graph.edge_end(src, Galois::MethodFlag::UNPROTECTED));
      PNode& ddata = graph.getData(dst, Galois::MethodFlag::UNPROTECTED);
      float delta = sdata.value / neighbors;
      
      ddata.accum.atomicIncrement(delta);
      return false; // Topology-driven
    }
  };

  struct UpdateNode {
    LigraAlgo* self;
    Graph& graph;
    UpdateNode(LigraAlgo* s, Graph& g): self(s), graph(g) { }
    void operator()(GNode src) const {
      PNode& sdata = graph.getData(src, Galois::MethodFlag::UNPROTECTED);
      float value = (1.0 - alpha) * sdata.accum.read() + alpha;
      float diff = std::fabs(value - sdata.value);
      if (diff <= tolerance)
        self->small_delta += 1;
      self->max_delta.update(diff);
      self->sum_delta.update(diff);
      sdata.value = value;
      sdata.accum.write(0);
    }
  };

  void operator()(Graph& graph) { 
    Galois::GraphNodeBagPair<> bags(graph.size());

    unsigned iteration = 0;

    // Initialize
    this->outEdgeMap(memoryLimit, graph, EdgeOperator(), bags.next());
    Galois::do_all_local(graph, UpdateNode(this, graph));

    while (true) {
      iteration += 1;
      float delta = max_delta.reduce();
      size_t sdelta = small_delta.reduce();
      std::cout << "iteration: " << iteration
                << " sum delta: " << sum_delta.reduce()
                << " max delta: " << delta
                << " small delta: " << sdelta
                << " (" << sdelta / (float) graph.size() << ")"
                << "\n";
      if (delta <= tolerance || iteration >= maxIterations)
        break;
      max_delta.reset();
      small_delta.reset();
      sum_delta.reset();
      //bags.swap();

      //this->outEdgeMap(memoryLimit, graph, EdgeOperator(), bags.cur(), bags.next(), true);
      this->outEdgeMap(memoryLimit, graph, EdgeOperator(), bags.next());
      Galois::do_all_local(graph, UpdateNode(this, graph));
    }
    
    if (iteration >= maxIterations) {
      std::cout << "Failed to converge\n";
    }
  }
};

#endif
