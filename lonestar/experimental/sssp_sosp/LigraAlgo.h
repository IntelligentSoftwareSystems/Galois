#ifndef APPS_SSSP_LIGRAALGO_H
#define APPS_SSSP_LIGRAALGO_H

#include "galois/DomainSpecificExecutors.h"
#include "galois/graphs/OCGraph.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/GraphNodeBag.h"

#include <boost/mpl/if.hpp>

#include "SSSP.h"

template <bool UseGraphChi>
struct LigraAlgo : public galois::ligraGraphChi::ChooseExecutor<UseGraphChi> {
  struct LNode : public SNode {
    bool visited;
  };

  typedef typename galois::graphs::LC_InlineEdge_Graph<LNode, uint32_t>::
      template with_compressed_node_ptr<true>::type ::template with_no_lockable<
          true>::type ::template with_numa_alloc<true>::type InnerGraph;
  typedef typename boost::mpl::if_c<
      UseGraphChi, galois::graphs::OCImmutableEdgeGraph<LNode, uint32_t>,
      galois::graphs::LC_InOut_Graph<InnerGraph>>::type Graph;
  typedef typename Graph::GraphNode GNode;

  std::string name() const { return UseGraphChi ? "LigraChi" : "Ligra"; }

  void readGraph(Graph& graph) {
    readInOutGraph(graph);
    this->checkIfInMemoryGraph(graph, memoryLimit);
  }

  struct Initialize {
    Graph& graph;
    Initialize(Graph& g) : graph(g) {}
    void operator()(GNode n) const {
      LNode& data  = graph.getData(n);
      data.dist    = DIST_INFINITY;
      data.visited = false;
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
                    typename GTy::edge_data_reference weight) {
      LNode& sdata = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      LNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

      while (true) {
        Dist newDist = sdata.dist + weight;
        Dist oldDist = ddata.dist;
        if (oldDist <= newDist)
          return false;
        if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
          return __sync_bool_compare_and_swap(&ddata.visited, false, true);
        }
      }
      return false;
    }
  };

  struct ResetVisited {
    Graph& graph;
    ResetVisited(Graph& g) : graph(g) {}
    void operator()(size_t n) const {
      graph.getData(graph.nodeFromId(n)).visited = false;
    }
  };

  void operator()(Graph& graph, const GNode& source) {
    galois::Statistic roundStat("Rounds");

    galois::graphsNodeBagPair<> bags(graph.size());

    graph.getData(source).dist = 0;

    this->outEdgeMap(memoryLimit, graph, EdgeOperator(), source, bags.next());
    galois::do_all(bags.next(), ResetVisited(graph));

    unsigned rounds = 0;
    while (!bags.next().empty()) {
      if (++rounds == graph.size()) {
        std::cout << "Negative weight cycle\n";
        break;
      }

      bags.swap();
      this->outEdgeMap(memoryLimit, graph, EdgeOperator(), bags.cur(),
                       bags.next(), true);
      galois::do_all(bags.next(), ResetVisited(graph));
    }

    roundStat += rounds + 1;
  }
};

#endif
