#ifndef APPS_BFS_LIGRAALGO_H
#define APPS_BFS_LIGRAALGO_H

#include "galois/DomainSpecificExecutors.h"
#include "galois/graphs/OCGraph.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/GraphNodeBag.h"

#include <boost/mpl/if.hpp>

#include "BFS.h"

template <bool UseGraphChi>
struct LigraBFS : public galois::ligraGraphChi::ChooseExecutor<UseGraphChi> {
  typedef typename galois::graphs::LC_CSR_Graph<SNode, void>::
      template with_no_lockable<true>::type ::template with_numa_alloc<
          true>::type InnerGraph;
  typedef typename boost::mpl::if_c<
      UseGraphChi, galois::graphs::OCImmutableEdgeGraph<SNode, void>,
      galois::graphs::LC_InOut_Graph<InnerGraph>>::type Graph;
  typedef typename Graph::GraphNode GNode;

  std::string name() const { return UseGraphChi ? "LigraChi" : "Ligra"; }

  void readGraph(Graph& graph) {
    readInOutGraph(graph);
    this->checkIfInMemoryGraph(graph, memoryLimit);
  }

  struct EdgeOperator {
    Dist newDist;
    EdgeOperator(Dist d) : newDist(d) {}

    template <typename GTy>
    bool cond(GTy& graph, typename GTy::GraphNode n) {
      return graph.getData(n, galois::MethodFlag::UNPROTECTED).dist ==
             DIST_INFINITY;
    }

    template <typename GTy>
    bool operator()(GTy& graph, typename GTy::GraphNode src,
                    typename GTy::GraphNode dst,
                    typename GTy::edge_data_reference) {
      SNode& ddata = graph.getData(dst, galois::MethodFlag::UNPROTECTED);

      Dist oldDist;
      while (true) {
        oldDist = ddata.dist;
        if (oldDist <= newDist)
          return false;
        if (__sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist)) {
          return true;
        }
      }
      return false;
    }
  };

  void operator()(Graph& graph, const GNode& source) {
    galois::graphsNodeBagPair<> bags(graph.size());

    Dist newDist               = 1;
    graph.getData(source).dist = 0;

    this->outEdgeMap(memoryLimit, graph, EdgeOperator(newDist), source,
                     bags.next());

    while (!bags.next().empty()) {
      bags.swap();
      newDist++;
      this->outEdgeMap(memoryLimit, graph, EdgeOperator(newDist), bags.cur(),
                       bags.next(), false);
    }
  }
};

template <bool UseGraphChi>
struct LigraDiameter
    : public galois::LigraGraphChi::ChooseExecutor<UseGraphChi> {
  typedef int Visited;

  struct LNode : public SNode {
    Visited visited[2];
  };

  typedef typename galois::graphs::LC_CSR_Graph<LNode, void>::
      template with_no_lockable<true>::type ::template with_numa_alloc<
          true>::type InnerGraph;
  typedef typename boost::mpl::if_c<
      UseGraphChi, galois::graphs::OCImmutableEdgeGraph<LNode, void>,
      galois::graphs::LC_InOut_Graph<InnerGraph>>::type Graph;
  typedef typename Graph::GraphNode GNode;

  void readGraph(Graph& graph) {
    readInOutGraph(graph);
    this->checkIfInMemoryGraph(graph, memoryLimit);
  }

  struct Initialize {
    Graph& graph;
    Initialize(Graph& g) : graph(g) {}
    void operator()(GNode n) const {
      LNode& data     = graph.getData(n, galois::MethodFlag::UNPROTECTED);
      data.dist       = DIST_INFINITY;
      data.visited[0] = data.visited[1] = 0;
    }
  };

  struct EdgeOperator {
    LigraDiameter* self;
    int cur;
    int next;
    Dist newDist;
    EdgeOperator(LigraDiameter* s, int c, int n, Dist d)
        : self(s), cur(c), next(n), newDist(d) {}

    template <typename GTy>
    bool cond(GTy& graph, typename GTy::GraphNode) {
      return true;
    }

    template <typename GTy>
    bool operator()(GTy& graph, typename GTy::GraphNode src,
                    typename GTy::GraphNode dst,
                    typename GTy::edge_data_reference) {
      LNode& sdata    = graph.getData(src, galois::MethodFlag::UNPROTECTED);
      LNode& ddata    = graph.getData(dst, galois::MethodFlag::UNPROTECTED);
      Visited toWrite = sdata.visited[cur] | ddata.visited[cur];

      if (toWrite != ddata.visited[cur]) {
        while (true) {
          Visited old  = ddata.visited[next];
          Visited newV = old | toWrite;
          if (old == newV)
            break;
          if (__sync_bool_compare_and_swap(&ddata.visited[next], old, newV))
            break;
        }
        Dist oldDist = ddata.dist;
        if (ddata.dist != newDist)
          return __sync_bool_compare_and_swap(&ddata.dist, oldDist, newDist);
      }
      return false;
    }
  };

  struct Update {
    LigraDiameter* self;
    Graph& graph;
    int cur;
    int next;
    Update(LigraDiameter* s, Graph& g, int c, int n)
        : self(s), graph(g), cur(c), next(n) {}
    void operator()(size_t id) const {
      LNode& data =
          graph.getData(graph.nodeFromId(id), galois::MethodFlag::UNPROTECTED);
      data.visited[next] |= data.visited[cur];
    }
  };

  size_t operator()(Graph& graph, const GNode& source) {
    galois::graphsNodeBagPair<> bags(graph.size());

    if (graph.size() && *graph.begin() != source)
      std::cerr << "Warning: Ignoring user-requested start node\n";

    Dist newDist        = 0;
    unsigned sampleSize = std::min(graph.size(), sizeof(Visited) * 8);
    unsigned count      = 0;
    for (typename Graph::iterator ii = graph.begin(), ei = graph.end();
         ii != ei; ++ii) {
      LNode& data     = graph.getData(*ii);
      data.dist       = 0;
      data.visited[1] = (Visited)1 << count;
      bags.next().push(graph.idFromNode(*ii), graph.size());

      if (++count >= sampleSize)
        break;
    }

    while (!bags.next().empty()) {
      bags.swap();
      newDist++;
      int cur  = newDist & 1;
      int next = (newDist + 1) & 1;
      galois::do_all(bags.cur(), Update(this, graph, cur, next));
      this->outEdgeMap(memoryLimit, graph,
                       EdgeOperator(this, cur, next, newDist), bags.cur(),
                       bags.next(), false);
    }

    return newDist - 1;
  }
};

#endif
