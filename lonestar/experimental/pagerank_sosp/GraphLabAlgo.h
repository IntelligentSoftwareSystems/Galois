#ifndef APPS_PAGERANK_GRAPHLABALGO_H
#define APPS_PAGERANK_GRAPHLABALGO_H

#include "galois/DomainSpecificExecutors.h"
#include "galois/Graphs/OCGraph.h"
#include "galois/Graphs/LCGraph.h"
#include "galois/Graphs/GraphNodeBag.h"

#include <boost/mpl/if.hpp>

#include "PageRankOld.h"

template<bool UseDelta, bool UseAsync>
struct GraphLabAlgo {
  struct LNode {
    float data;
    float getPageRank() { return data; }
  };

  typedef typename galois::graphs::LC_CSR_Graph<LNode,void>
    ::template with_numa_alloc<true>::type
    ::template with_no_lockable<true>::type
    InnerGraph;
  typedef galois::graphs::LC_InOut_Graph<InnerGraph> Graph;
  typedef typename Graph::GraphNode GNode;

  std::string name() const { return "GraphLab"; }

  void readGraph(Graph& graph) {
    // Using dense forward option, so we don't need in-edge information
    galois::graphs::readGraph(graph, filename); 
  }

  struct Initialize {
    Graph& g;
    Initialize(Graph& g): g(g) { }
    void operator()(typename Graph::GraphNode n) const {
      LNode& data = g.getData(n, galois::MethodFlag::UNPROTECTED);
      data.data = 1.0;
    }
  };

  template<bool UseD>
  struct Program {
    struct gather_type {
      float data;
      gather_type(): data(0) { }
    };

    typedef galois::graphsLab::EmptyMessage message_type;

    typedef int tt_needs_gather_in_edges;
    typedef int tt_needs_scatter_out_edges;

    float last_change;

    void gather(Graph& graph, GNode node, GNode src, GNode dst, gather_type& sum, typename Graph::edge_data_reference) { 
      int outs = std::distance(graph.edge_begin(src, galois::MethodFlag::UNPROTECTED),
          graph.edge_end(src, galois::MethodFlag::UNPROTECTED));
      sum.data += graph.getData(src, galois::MethodFlag::UNPROTECTED).data / outs;
    }
    
    void init(Graph& graph, GNode node, const message_type& msg) { }

    void apply(Graph& graph, GNode node, const gather_type& total) {
      LNode& data = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      int outs = std::distance(graph.edge_begin(node, galois::MethodFlag::UNPROTECTED),
          graph.edge_end(node, galois::MethodFlag::UNPROTECTED));
      float newval = (1.0 - alpha) * total.data + alpha;
      last_change = (newval - data.data) / outs;
      data.data = newval;
    }

    bool needsScatter(Graph& graph, GNode node) {
      if (UseD)
        return std::fabs(last_change) > tolerance;
      return false;
    }

    void scatter(Graph& graph, GNode node, GNode src, GNode dst,
        galois::graphsLab::Context<Graph,Program>& ctx, typename Graph::edge_data_reference) {
      ctx.push(dst, message_type());
    }
  };

  void operator()(Graph& graph) {
    if (UseAsync) {
      // Asynchronous execution
      galois::graphsLab::AsyncEngine<Graph,Program<true> > engine(graph, Program<true>());
      engine.execute();
    } else if (UseDelta) {
      galois::graphsLab::SyncEngine<Graph,Program<true> > engine(graph, Program<true>());
      engine.execute();
    } else {
      galois::graphsLab::SyncEngine<Graph,Program<false> > engine(graph, Program<false>());
      for (unsigned i = 0; i < maxIterations; ++i)
        engine.execute();
    }
  }
};

#endif
